import re
import json
import html
import time
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st

import pymongo
import mongomock
import fakeredis

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from detoxify import Detoxify
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Page Config & Custom CSS Styling
st.set_page_config(page_title="ConversationGuardian - Moderator Command Center", layout="wide")

st.markdown("""
<style>
    .chat-card-toxic { background-color: #321c1c; border-left: 5px solid #e53e3e; padding: 12px; border-radius: 6px; margin-bottom: 10px; color: #fff5f5; }
    .chat-card-policy { background-color: #322d1c; border-left: 5px solid #d69e2e; padding: 12px; border-radius: 6px; margin-bottom: 10px; color: #fffaf0; }
    .chat-card-civil { background-color: #1c3226; border-left: 5px solid #38a169; padding: 12px; border-radius: 6px; margin-bottom: 10px; color: #f0fff4; }
    .chat-card-neu { background-color: #1a202c; border-left: 5px solid #a0aec0; padding: 12px; border-radius: 6px; margin-bottom: 10px; color: #ffffff; }
    .meta-tag { font-size: 0.8rem; opacity: 0.85; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# 2. Paths
PROJECT_DIR = Path(__file__).parent.parent.resolve()
if not (PROJECT_DIR / "models").exists():
    PROJECT_DIR = Path(__file__).parent.parent.resolve()

DATA_DIR = PROJECT_DIR / "data"
CLEANED_DIR = DATA_DIR / "cleaned"
FEATURES_DIR = DATA_DIR / "features"
SEQUENCES_DIR = DATA_DIR / "sequences"
PATTERNS_DIR = DATA_DIR / "patterns"
MODEL_DIR = "Nayanojwal/conversation-guardian-model"

POLICY_REGEX = re.compile(r'\b(WP|Wikipedia):[A-Z0-9_-]+\b', re.IGNORECASE)

# --- Decision engine constants (documented, not hand-waved) ---
# TODO: replace these two with the real tertile cutoffs computed from your
# validation set's risk_score distribution (np.percentile(val_scores, [33, 66])),
# per REFERENCE_GUIDE.md. Placeholders below are reasonable defaults only.
HIGH_RISK_THRESHOLD = 0.65
MODERATE_RISK_THRESHOLD = 0.45
MIN_HEALTHY_SUPPORT_FOR_TRUSTED_RATIO = 5  # patterns below this show as "weak evidence", not a headline multiplier

@st.cache_resource
def load_all():
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    vader = SentimentIntensityAnalyzer()
    detoxify_model = Detoxify('original', device=device.type)

    with open(PATTERNS_DIR / "mined_patterns.json", "r", encoding="utf-8") as f:
        mined_patterns = json.load(f)

    mongo_client = mongomock.MongoClient()
    db = mongo_client["ConversationGuardian"]

    db["mined_patterns"].drop()
    db["mined_patterns"].insert_many([{**p, "_id": str(p["pattern_id"])} for p in mined_patterns])

    train_cleaned = pd.read_parquet(CLEANED_DIR / "train_cleaned.parquet")
    train_feats = pd.read_parquet(FEATURES_DIR / "train_features.parquet")
    train_seqs = pd.read_parquet(SEQUENCES_DIR / "train_sequences.parquet")
    seq_map = train_seqs.set_index('conversation_id')['symbol_sequence'].to_dict()

    merged = pd.merge(
        train_cleaned,
        train_feats[['conversation_id', 'utterance_index', 'sentiment_compound', 'toxicity']],
        on=['conversation_id', 'utterance_index'],
        how='inner'
    ).sort_values(['conversation_id', 'utterance_index'])

    docs = []
    for conv_id, group in merged.groupby('conversation_id'):
        docs.append({
            "_id": f"conv_{conv_id}",
            "conversation_id": str(conv_id),
            "split": str(group['split'].iloc[0]),
            "has_attack": bool(group['has_attack'].iloc[0]),
            "symbol_sequence": [str(s) for s in list(seq_map.get(conv_id, []))],
            "utterances": [
                {
                    "speaker": str(row['speaker']),
                    "clean_text": str(row['clean_text']),
                    "toxicity": round(float(row['toxicity']), 4)
                } for _, row in group.iterrows()
            ]
        })

    db["conversations"].drop()
    db["conversations"].insert_many(docs)
    redis_client = fakeredis.FakeStrictRedis()

    return model, tokenizer, vader, detoxify_model, mined_patterns, db, redis_client, device

model, tokenizer, vader, detoxify_model, mined_patterns, db, redis_client, device = load_all()

# 3. Helper Functions
def clean_wikipedia_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip(): return "[EMPTY_MESSAGE]"
    if "&" in text: text = html.unescape(text)
    text = re.sub(r'~~~+', '', text)
    text = re.sub(r'\{\{[\s\S]*?\}\}', '', text)
    text = re.sub(r'\[\[[^\]|]*\|([^\]]+)\]\]', r'\1', text)
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'(?m)^\:+[ \t]*', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '<URL>', text)
    res = re.sub(r'[ \t]+', ' ', text).strip()
    return res if res else "[EMPTY_MESSAGE]"

def map_state_locked(sentiment: float, toxicity: float, has_policy_cite: bool, threshold: float = 0.3597) -> str:
    is_neg = sentiment <= -0.05
    is_pos = sentiment >= 0.05
    is_toxic = toxicity >= threshold

    if is_toxic and is_neg: return "NEGATIVE_TOXIC"
    elif has_policy_cite and is_neg: return "NEGATIVE_POLICY"
    elif is_neg: return "NEGATIVE"
    elif is_pos: return "POSITIVE"
    else: return "NEUTRAL"

def is_prefix_match(symbol_prefix: list, mined_pattern: list) -> bool:
    if len(symbol_prefix) >= len(mined_pattern) or all(s == "POSITIVE" for s in symbol_prefix):
        return False
    return mined_pattern[:len(symbol_prefix)] == symbol_prefix

def find_pattern_match(symbol_prefix: list, patterns_list: list) -> dict:
    """Among matching patterns, prefer the one with the most TRUSTWORTHY evidence
    (highest healthy_support), not just the highest (possibly artifact-inflated) ratio."""
    candidates = [p for p in patterns_list if is_prefix_match(symbol_prefix, p['pattern'])]
    if not candidates:
        return None
    trusted = [p for p in candidates if int(p.get('healthy_support', 0)) >= MIN_HEALTHY_SUPPORT_FOR_TRUSTED_RATIO]
    pool = trusted if trusted else candidates
    return max(pool, key=lambda p: float(p['support_ratio']))

def analyze_conversation(messages: list) -> dict:
    clean_msgs = [clean_wikipedia_text(m) for m in messages if isinstance(m, str) and m.strip()]
    if not clean_msgs: clean_msgs = ["[EMPTY_MESSAGE]"]

    states, toxic_scores, msg_details, found_cites = [], [], [], []
    has_policy_cite = False
    chart_data = []

    for idx, msg in enumerate(clean_msgs):
        s_score = float(vader.polarity_scores(msg)['compound']) if msg != "[EMPTY_MESSAGE]" else 0.0
        t_score = float(detoxify_model.predict([msg])['toxicity'][0]) if msg != "[EMPTY_MESSAGE]" else 0.0

        cites = POLICY_REGEX.findall(msg)
        is_policy = len(cites) > 0
        if is_policy:
            has_policy_cite = True
            found_cites.extend(cites)

        st_val = map_state_locked(s_score, t_score, is_policy)
        states.append(st_val)
        toxic_scores.append(t_score)

        chart_data.append({"Message": f"Msg {idx+1}", "Sentiment": s_score, "Toxicity": t_score})

        if "TOXIC" in st_val: card_cls = "chat-card-toxic"
        elif "POLICY" in st_val: card_cls = "chat-card-policy"
        elif "POSITIVE" in st_val: card_cls = "chat-card-civil"
        else: card_cls = "chat-card-neu"

        msg_details.append({
            "index": idx + 1, "text": msg, "state": st_val,
            "toxicity": t_score, "sentiment": s_score, "card_class": card_cls
        })

    text_prefix = " [SEP] ".join(clean_msgs)
    inputs = tokenizer(text_prefix, truncation=True, max_length=256, padding="max_length", return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        risk_score = float(torch.softmax(logits, dim=1)[0][1].cpu().numpy())

    matched_pat = find_pattern_match(states, mined_patterns)
    max_toxicity = max(toxic_scores) if toxic_scores else 0.0

    # =========================================================================
    # DECISION ENGINE
    # -------------------------------------------------------------------------
    # `decision` (the verdict banner) is a function of risk_score and
    # max_toxicity ONLY. Policy citations and pattern matches are informational
    # context added to `explanation` - they can NEVER change the verdict.
    # This was previously an override (policy citations downgraded high-risk
    # scores to "Moderate"); an ablation study showed that override reduced
    # recall by 14 points, so it has been removed. Do not reintroduce it.
    # =========================================================================
    pattern_is_trustworthy = (
        matched_pat is not None
        and int(matched_pat.get('healthy_support', 0)) >= MIN_HEALTHY_SUPPORT_FOR_TRUSTED_RATIO
    )

    if max_toxicity >= 0.3597:
        decision = "HIGH RISK (PERSONAL ATTACK DETECTED)"
        explanation = (
            f"Personal Attack Detected: Toxic language detected in message text (Peak Toxicity: {max_toxicity:.1%}).\n\n"
            f"Recommended Moderator Action: Remove toxic comment immediately and issue user warning per WP:CIVIL guidelines."
        )
    elif risk_score >= HIGH_RISK_THRESHOLD:
        decision = "HIGH RISK (MODEL-PREDICTED ESCALATION)"
        explanation = (
            f"High Risk Predicted: DistilBERT model predicts a {risk_score:.1%} derailment probability "
            f"from text content alone (no toxic language detected yet - this is an early warning).\n\n"
            f"Recommended Moderator Action: Flag thread for moderator review."
        )
    elif risk_score >= MODERATE_RISK_THRESHOLD:
        decision = "MODERATE RISK"
        explanation = (
            f"Moderate Risk: DistilBERT predicts a {risk_score:.1%} derailment probability - "
            f"elevated but not high-confidence.\n\n"
            f"Recommended Moderator Action: Monitor thread."
        )
    else:
        decision = "LOW RISK (HEALTHY CONVERSATION)"
        explanation = (
            f"Healthy Conversation: DistilBERT risk score is {risk_score:.1%}. Language remains civil.\n\n"
            f"Recommended Moderator Action: No moderator intervention required."
        )

    # --- Informational context only - never changes `decision` above ---
    if has_policy_cite:
        cites_str = ", ".join(set(found_cites))
        explanation += (
            f"\n\nContext: Wikipedia policy citations detected ({cites_str}). "
            f"This is shown for context only and does not affect the risk tier - "
            f"our ablation study found policy citations do not reliably predict lower risk."
        )

    if pattern_is_trustworthy:
        explanation += (
            f"\n\nHistorical evidence: sequence matches pattern '{matched_pat['pattern_str']}' "
            f"({matched_pat['derailed_support']} derailed vs {matched_pat['healthy_support']} healthy past cases, "
            f"{matched_pat['support_ratio']}x multiplier)."
        )
    elif matched_pat:
        explanation += (
            f"\n\nWeak historical match: pattern '{matched_pat['pattern_str']}' has too few healthy-case "
            f"observations ({matched_pat.get('healthy_support', 0)}) for a statistically reliable multiplier - "
            f"shown for reference only, not used in the verdict."
        )

    return {
        "risk_score": risk_score, "decision": decision, "states": states,
        "matched_pattern": matched_pat, "pattern_is_trustworthy": pattern_is_trustworthy,
        "explanation": explanation, "msg_details": msg_details,
        "chart_data": pd.DataFrame(chart_data)
    }

# 4. Streamlit UI Layout
st.title("ConversationGuardian - Moderator Command Center")

tab1, tab2, tab3 = st.tabs(["Live Risk Simulator", "Pattern Catalog Explorer", "NoSQL Database Analytics"])

CATEGORIZED_PRESETS = {
    "Escalation (High Risk)": {
        "Edit War Escalation": "Please stop reverting my edits without giving a valid Wikipedia policy reason.\nYour edits are completely unsourced and violate NPOV guidelines!\nYou are an arrogant jerk and a vandal who does not understand Wikipedia rules!",
        "Talk Page Admin Dispute": "Please stop deleting valid sections from this talk page without community discussion.\nYour contributions are garbage hoaxes and you are intentionally vandalizing this article!\nYou are a pathetic power-tripping bully, an arrogant idiot, and a disgrace to Wikipedia!",
        "Reference Accusation": "I added reliable references to the article as requested.\nThose sources are fake news from self-published blogs, stop edit warring!\nYou are a dishonest hypocrite and an abusive editor!",
        "Consensus Dispute": "Check the talk page consensus before removing this paragraph.\nThere is no consensus for your disruptive original research, stop vandalizing!\nYou are an incompetent troll who should be banned from editing!",
        "Lead Section Edit War": "Why did you undo my changes to the lead section?\nBecause your changes are complete trash and show zero reading comprehension.\nYou are an insufferable idiot and a malicious admin puppet!"
    },
    "Normal (Healthy Discussion)": {
        "Policy & Source Discussion": "Please provide reliable secondary sources for these factual claims per WP:VERIFY policy.\nI added two university press footnotes in the references section to support the text.\nThanks for adding those exact page citations, it resolves the neutrality tag for this section.",
        "Timeline Expansion": "Should we add a section clarifying the historical timeline?\nYes, that sounds like a good idea if we cite standard academic texts.\nI will draft a short paragraph on the talk page for review first.",
        "Formatting Cleanup": "Thanks for fixing the typo in the footnote citation.\nNo problem, happy to help clean up the reference formatting.\nAppreciate your help keeping the article looking polished.",
        "Guideline Linking": "I added a link to the relevant project guidelines page.\nThat is very helpful context for new editors joining the discussion.\nGlad it helped, let me know if any other sections need review.",
        "Lead Summary Consensus": "Let's make sure the intro summary covers all major sub-sections.\nAgreed, I will rephrase the second paragraph to reflect the consensus.\nLooks much clearer now, thanks for working together on this."
    },
    "Medium (Firm Policy Debate)": {
        "Fringe Science / WP:RS Debate": "This source is from a self-published blog and fails the WP:RS criteria for medical claims.\nAre you calling my sources fake news? You are deliberately suppressing facts you dislike!\nNo one is suppressing facts, we are applying the WP:WEIGHT policy to minority opinions.",
        "3RR Warning / Edit Warring": "You keep reverting my edits without giving any policy explanation in your edit summary.\nBecause your changes violate established consensus per WP:NPOV guidelines!\nStop reverting me or I will report your disruptive edit warring to the noticeboard under WP:3RR.",
        "BRD Cycle Discussion": "I do not agree with renaming this section header without a formal talk page proposal.\nPlease read the BRD cycle guideline WP:BRD before undoing the new section title.\nFair enough, I see the community consensus from the previous discussion archive now.",
        "Block Warning Dispute": "WARNING: Your recent edits violate WP:3RR and WP:EDITWAR policies immediately!\nI am well aware of WP:3RR and I have only made two reverts in the last 24 hours.\nCheck the timestamp history again, you are on your third revert and on the verge of a block!",
        "Lead Summary Policy": "I disagree with removing this paragraph from the lead section.\nPer WP:LEAD, the intro should only summarize body content already present in the article.\nUnderstood, let's expand the body section first before re-adding it to the lead."
    }
}

with tab1:
    col_input, col_results = st.columns([1.1, 1], gap="large")

    with col_input:
        st.subheader("Input Thread Simulator")

        input_mode = st.radio("Choose Input Method:", ["Scenario Presets", "Custom Typing (Multiline)"], horizontal=True)

        user_lines = []
        if input_mode == "Scenario Presets":
            category = st.selectbox("Select Scenario Category:", list(CATEGORIZED_PRESETS.keys()))
            scenario = st.selectbox("Select Wikipedia Scenario:", list(CATEGORIZED_PRESETS[category].keys()))
            preset_text = CATEGORIZED_PRESETS[category][scenario]
            raw_input = st.text_area("Selected Preset Conversation Text:", value=preset_text, height=180)
            user_lines = [l.strip() for l in raw_input.split("\n") if l.strip()]
        else:
            raw_input = st.text_area("Type custom conversation (One message per line):", value="User A: Thanks for your edit.\nUser B: I completely disagree with your removal of my section!\nUser A: You are an arrogant jerk and a vandal!", height=200)
            user_lines = [l.strip() for l in raw_input.split("\n") if l.strip()]

        run_btn = st.button("Run Derailment Risk Analysis", type="primary", use_container_width=True)

        if run_btn or user_lines:
            res = analyze_conversation(user_lines)

            st.markdown("#### Turn-by-Turn Moderator Chat Feed")
            for item in res["msg_details"]:
                st.markdown(
                    f'<div class="{item["card_class"]}">'
                    f'<b>Turn {item["index"]}:</b> {item["text"]}<br/>'
                    f'<div class="meta-tag">State Tag: <code>{item["state"]}</code> | Toxicity: <b>{item["toxicity"]:.1%}</b> | Sentiment: <b>{item["sentiment"]:+.2f}</b></div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    with col_results:
        st.subheader("Moderator Intelligence Board")
        if run_btn or user_lines:
            m1, m2, m3 = st.columns(3)
            m1.metric("DistilBERT Risk Score", f"{res['risk_score']:.1%}")
            m2.metric("Matched Pattern ID", str(res['matched_pattern']['pattern_id']) if res['matched_pattern'] else "None")
            if res['matched_pattern']:
                ratio_label = f"{float(res['matched_pattern']['support_ratio']):.1f}x"
                if not res['pattern_is_trustworthy']:
                    ratio_label += " (weak evidence)"
                m3.metric("Risk Multiplier", ratio_label)
            else:
                m3.metric("Risk Multiplier", "1.0x")

            if "HIGH RISK" in res['decision']: st.error(f"**Verdict:** {res['decision']}")
            elif "MODERATE" in res['decision']: st.warning(f"**Verdict:** {res['decision']}")
            else: st.success(f"**Verdict:** {res['decision']}")

            st.write(f"**State Trajectory Flow:** `{' -> '.join(res['states'])}`")
            st.info(res['explanation'])

            st.markdown("#### Turn-by-Turn Sentiment vs Toxicity")
            st.line_chart(res['chart_data'].set_index("Message"))

            if res['matched_pattern']:
                st.subheader("Ground-Truth Wikipedia Evidence (MongoDB)")
                e_ids = [str(i) for i in res['matched_pattern'].get('representative_conversations', [])]
                if e_ids:
                    s_id = st.selectbox("Select Wikipedia Thread ID:", e_ids)
                    doc = db["conversations"].find_one({"conversation_id": s_id})
                    if doc:
                        with st.expander(f"View Real Wikipedia Thread Document ({s_id})", expanded=True):
                            for u in doc.get("utterances", []):
                                st.write(f"**[{u['speaker']}]**: {u['clean_text']} *(Toxicity: {u['toxicity']:.2f})*")

with tab2:
    st.subheader(f"PrefixSpan Pattern Catalog Explorer ({len(mined_patterns)} Mined Trajectories)")
    st.markdown("Searchable catalog of all leak-free escalation trajectories mined from Wikipedia history.")
    st.caption(
        f"Note: patterns with fewer than {MIN_HEALTHY_SUPPORT_FOR_TRUSTED_RATIO} healthy-case observations "
        f"have statistically unstable support ratios - treat their multiplier as illustrative, not precise."
    )
    p_df = pd.DataFrame(mined_patterns)[['pattern_id', 'pattern_str', 'support_ratio', 'derailed_support', 'healthy_support']]
    p_df['ratio_reliable'] = p_df['healthy_support'] >= MIN_HEALTHY_SUPPORT_FOR_TRUSTED_RATIO
    st.dataframe(p_df, use_container_width=True, height=480)

with tab3:
    st.subheader("Interactive NoSQL Database Workbench (MongoDB)")

    st.markdown("#### Live MongoDB Document Search")
    query_speaker = st.text_input("Filter MongoDB Conversations by Speaker Name:", value="Jytdog")
    if query_speaker:
        matching_docs = list(db["conversations"].find({"utterances.speaker": query_speaker}).limit(1))
        st.write(f"Found matching conversation documents in MongoDB for speaker `{query_speaker}`. Note the Embedded Schema pattern:")
        for d in matching_docs:
            d.pop("_id", None)
            st.json(d)

    st.markdown("---")
    st.markdown("#### MongoDB `$unwind` Aggregation Pipeline Analytics")
    pipeline = [
        {"$unwind": "$utterances"},
        {"$match": {"utterances.toxicity": {"$gte": 0.3597}}},
        {"$group": {"_id": "$utterances.speaker", "toxic_comment_count": {"$sum": 1}}},
        {"$sort": {"toxic_comment_count": -1}},
        {"$limit": 5}
    ]
    top_speakers_df = pd.DataFrame(list(db["conversations"].aggregate(pipeline))).rename(columns={"_id": "Speaker Name", "toxic_comment_count": "Toxic Comments Count"})
    st.dataframe(top_speakers_df, use_container_width=True)
