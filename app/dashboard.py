import hashlib
import html
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from dotenv import load_dotenv
from detoxify import Detoxify
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

try:
    from app.database import connect_cache, connect_database, seed_reference_data
except ModuleNotFoundError:
    # Supports launching Streamlit from inside the app/ directory.
    from database import connect_cache, connect_database, seed_reference_data

# 1. Page Config & Custom CSS Styling
st.set_page_config(page_title="ConversationGuardian - Moderator Command Center", layout="wide")

st.markdown("""
<style>
    :root { --cg-ink: #e6edf7; --cg-muted: #9aa9bd; --cg-border: rgba(148,163,184,.22); }
    .block-container { max-width: 1450px; padding-top: 2rem; padding-bottom: 3rem; }
    [data-testid="stAppViewContainer"] { background: radial-gradient(ellipse at top left, rgba(42,76,122,.24), transparent 42%), #0b1120; }
    [data-testid="stHeader"] { background: transparent; }
    h1, h2, h3 { letter-spacing: -.025em; }
    .cg-eyebrow { text-transform: uppercase; letter-spacing: .15em; font-size: .72rem; font-weight: 700; color: #7dd3fc; margin-bottom: .35rem; }
    .cg-subtitle { color: #aebbd0; font-size: 1rem; line-height: 1.6; max-width: 850px; margin-bottom: 1rem; }
    .cg-panel { border: 1px solid var(--cg-border); border-radius: 16px; padding: 1rem 1.15rem; background: rgba(18,28,47,.72); margin: .35rem 0 1rem; }
    .cg-panel-title { font-weight: 700; font-size: 1rem; margin-bottom: .25rem; }
    .cg-panel-caption { color: #aebbd0; font-size: .86rem; line-height: 1.5; }
    .chat-card-toxic { background-color: #321c1c; border-left: 5px solid #ef4444; padding: 14px; border-radius: 9px; margin-bottom: 10px; color: #fff5f5; overflow-wrap: anywhere; }
    .chat-card-policy { background-color: #322d1c; border-left: 5px solid #fbbf24; padding: 14px; border-radius: 9px; margin-bottom: 10px; color: #fffaf0; overflow-wrap: anywhere; }
    .chat-card-civil { background-color: #17352c; border-left: 5px solid #34d399; padding: 14px; border-radius: 9px; margin-bottom: 10px; color: #f0fff4; overflow-wrap: anywhere; }
    .chat-card-neu { background-color: #182337; border-left: 5px solid #94a3b8; padding: 14px; border-radius: 9px; margin-bottom: 10px; color: #f8fafc; overflow-wrap: anywhere; }
    .meta-tag { font-size: .8rem; opacity: .9; margin-top: 7px; color: #d1d9e6; }
    div[data-testid="stMetric"] { background: rgba(21,34,56,.72); border: 1px solid var(--cg-border); border-radius: 13px; padding: 12px 14px; }
    div[data-testid="stTabs"] button { font-weight: 650; }
    div[data-testid="stForm"] { border-color: var(--cg-border); }
    .stButton > button[kind="primary"] { border-radius: 10px; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# 2. Paths
PROJECT_DIR = Path(__file__).parent.parent.resolve()
load_dotenv(PROJECT_DIR / ".env", override=False)

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

def _setting_from_config(name: str) -> str | None:
    """Read connection settings from environment variables or Streamlit secrets."""
    value = os.getenv(name, "").strip()
    if not value:
        try:
            value = str(st.secrets.get(name, "")).strip()
        except Exception:
            value = ""
    return value or None


@st.cache_resource(show_spinner="Loading ConversationGuardian models and data...")
def load_all():
    import nltk

    try:
        nltk.download("vader_lexicon", quiet=True)
    except Exception:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    vader = SentimentIntensityAnalyzer()
    detoxify_model = Detoxify("original", device=device.type)

    with open(PATTERNS_DIR / "mined_patterns.json", "r", encoding="utf-8") as file:
        mined_patterns = json.load(file)

    train_cleaned = pd.read_parquet(CLEANED_DIR / "train_cleaned.parquet")
    train_feats = pd.read_parquet(FEATURES_DIR / "train_features.parquet")
    train_seqs = pd.read_parquet(SEQUENCES_DIR / "train_sequences.parquet")

    # Normalize identifier dtypes before joins/lookups; Parquet readers can infer
    # integer/object/string IDs differently across environments.
    for frame in (train_cleaned, train_feats, train_seqs):
        if "conversation_id" not in frame.columns:
            raise ValueError("Each prepared dataset must contain conversation_id.")
        frame["conversation_id"] = frame["conversation_id"].astype(str)

    seq_map = train_seqs.set_index("conversation_id")["symbol_sequence"].to_dict()

    merged = pd.merge(
        train_cleaned,
        train_feats[
            ["conversation_id", "utterance_index", "sentiment_compound", "toxicity"]
        ],
        on=["conversation_id", "utterance_index"],
        how="inner",
    ).sort_values(["conversation_id", "utterance_index"])

    docs = []
    for conv_id, group in merged.groupby("conversation_id"):
        docs.append(
            {
                "_id": f"conv_{conv_id}",
                "conversation_id": str(conv_id),
                "split": str(group["split"].iloc[0]),
                "has_attack": bool(group["has_attack"].iloc[0]),
                "symbol_sequence": [
                    str(state) for state in list(seq_map.get(conv_id, []))
                ],
                "utterances": [
                    {
                        "speaker": str(row["speaker"]),
                        "clean_text": str(row["clean_text"]),
                        "toxicity": round(float(row["toxicity"]), 4),
                    }
                    for _, row in group.iterrows()
                ],
            }
        )

    mongo_client, db, database_mode = connect_database(
        _setting_from_config("MONGODB_URI")
    )
    seed_reference_data(db, mined_patterns, docs)
    redis_client, redis_mode = connect_cache(_setting_from_config("REDIS_URL"))

    return (
        model,
        tokenizer,
        vader,
        detoxify_model,
        mined_patterns,
        db,
        redis_client,
        device,
        database_mode,
        redis_mode,
        mongo_client,
    )


try:
    (
        model,
        tokenizer,
        vader,
        detoxify_model,
        mined_patterns,
        db,
        redis_client,
        device,
        database_mode,
        redis_mode,
        mongo_client,
    ) = load_all()
except Exception as exc:
    logger.exception("ConversationGuardian startup failed")
    st.error(
        "ConversationGuardian could not finish loading. Check the app logs for "
        "model downloads, dataset files, or MongoDB connection problems."
    )
    st.caption(f"Startup failure type: {type(exc).__name__}")
    st.stop()

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

def toxicity_score_for_message(text: str) -> float:
    """Extract one toxicity score across Detoxify's scalar and sequence output forms."""
    prediction = detoxify_model.predict(text)
    if not isinstance(prediction, dict) or "toxicity" not in prediction:
        raise ValueError("Detoxify prediction did not contain a 'toxicity' score.")

    scores = np.asarray(prediction["toxicity"], dtype=float).reshape(-1)
    if scores.size != 1 or not np.isfinite(scores[0]):
        raise ValueError("Detoxify returned an invalid toxicity score for one message.")
    return float(scores[0])


def cached_toxicity_score(text: str) -> float:
    """Cache model scores by a message hash without storing the message in Redis."""
    cache_key = "convoguard:v1:toxicity:" + hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()
    try:
        cached = redis_client.get(cache_key)
        if cached is not None:
            redis_client.incr("convoguard:stats:toxicity_cache_hits")
            cached_score = float(cached)
            if np.isfinite(cached_score):
                return cached_score
    except Exception:
        # The app remains usable if an external cache has a transient failure.
        pass

    score = toxicity_score_for_message(text)
    try:
        redis_client.incr("convoguard:stats:toxicity_cache_misses")
        redis_client.set(cache_key, str(score), ex=86400)
    except Exception:
        pass
    return score


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
        t_score = cached_toxicity_score(msg) if msg != "[EMPTY_MESSAGE]" else 0.0

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
    # Policy citations and mined patterns are explanatory context only. Do not
    # change the verdict based on a policy-citation heuristic or an unsupported
    # claim about predictive value.
    # =========================================================================
    pattern_is_trustworthy = (
        matched_pat is not None
        and int(matched_pat.get('healthy_support', 0)) >= MIN_HEALTHY_SUPPORT_FOR_TRUSTED_RATIO
    )

    if max_toxicity >= 0.3597:
        decision = "HIGH RISK (TOXICITY FLAG — REVIEW REQUIRED)"
        explanation = (
            f"Toxicity flag: the toxicity model score reached {max_toxicity:.1%}. "
            f"Toxicity alone does not prove a targeted personal attack; context, quotation, "
            f"and the intended target matter.\n\n"
            f"Recommended Moderator Action: Review the flagged message and surrounding turns. "
            f"Do not remove a comment solely because of this score."
        )
    elif risk_score >= HIGH_RISK_THRESHOLD:
        decision = "HIGH MODEL SCORE (REVIEW REQUIRED)"
        explanation = (
            f"High model score: DistilBERT returned a {risk_score:.1%} softmax score for its positive class. "
            f"This score is not established as calibrated, and this full-thread simulator is not by itself "
            f"evidence of forecast accuracy.\n\n"
            f"Recommended Moderator Action: Flag the thread for human moderator review."
        )
    elif risk_score >= MODERATE_RISK_THRESHOLD:
        decision = "MODERATE MODEL SCORE"
        explanation = (
            f"Moderate model score: DistilBERT returned a {risk_score:.1%} softmax score for its positive class. "
            f"The score is not established as calibrated.\n\n"
            f"Recommended Moderator Action: Review context and monitor the thread."
        )
    else:
        decision = "LOW MODEL SCORE (NOT A SAFETY GUARANTEE)"
        explanation = (
            f"Lower model score: DistilBERT returned a {risk_score:.1%} softmax score for its positive class. "
            f"This does not prove the conversation is safe or that no attack will occur.\n\n"
            f"Recommended Moderator Action: No immediate flag from this model score; use moderator judgment."
        )

    # --- Informational context only - never changes `decision` above ---
    if has_policy_cite:
        cites_str = ", ".join(set(found_cites))
        explanation += (
            f"\n\nContext: Wikipedia policy citations detected ({cites_str}). "
            f"This is shown for context only and does not affect the risk tier. "
            f"A policy citation should not be treated as proof that a conversation is safe."
        )

    if pattern_is_trustworthy:
        explanation += (
            f"\n\nHistorical evidence: sequence matches pattern '{matched_pat['pattern_str']}' "
            f"({matched_pat['derailed_support']} attack-labelled vs {matched_pat['healthy_support']} healthy historical cases; "
            f"support ratio {matched_pat['support_ratio']}). This is descriptive historical evidence, not a causal effect "
            f"and does not multiply the DistilBERT score."
        )
    elif matched_pat:
        explanation += (
            f"\n\nWeak historical match: pattern '{matched_pat['pattern_str']}' has too few healthy-case "
            f"observations ({matched_pat.get('healthy_support', 0)}) for a stable ratio; "
            f"shown for reference only and not used in the verdict."
        )

    return {
        "risk_score": risk_score, "decision": decision, "states": states,
        "matched_pattern": matched_pat, "pattern_is_trustworthy": pattern_is_trustworthy,
        "explanation": explanation, "msg_details": msg_details,
        "chart_data": pd.DataFrame(chart_data)
    }

# 4. Streamlit UI Layout
st.markdown('<div class="cg-eyebrow">WIKIPEDIA DISCUSSION INTELLIGENCE</div>', unsafe_allow_html=True)
st.title("ConversationGuardian")
st.markdown(
    '<div class="cg-subtitle">Explore discussion risk, inspect mined escalation patterns, '
    'and demonstrate document-database operations in one moderation workspace.</div>',
    unsafe_allow_html=True,
)
if "mongomock" in database_mode:
    st.warning("Database is running in in-memory demo mode. Data will not persist after the app restarts. Configure MONGODB_URI to use a real MongoDB database.")
else:
    st.success("MongoDB is connected. Reference data is upserted; collections are not dropped.")
if "fakeredis" in redis_mode:
    st.caption("Redis cache: in-memory demo mode; cache entries disappear on restart.")
else:
    st.caption("Redis cache: persistent connection enabled.")

tab1, tab2, tab3, tab4 = st.tabs(["Risk Simulator", "Pattern Catalog", "NoSQL Lab", "Sequential Results"])

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
        st.caption(
            "This simulator scores the messages you provide. Use the Sequential Results tab for "
            "measured prefix-horizon comparisons; the full-thread model score is not itself a "
            "validated early-warning benchmark."
        )

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

        if run_btn:
            if not user_lines:
                st.warning("Enter at least one non-empty message before running analysis.")
                st.session_state.pop("analysis_result", None)
                st.session_state.pop("analysis_input", None)
            else:
                with st.spinner("Analyzing conversation..."):
                    try:
                        analysis_result = analyze_conversation(user_lines)
                    except Exception as exc:
                        logger.exception("Conversation analysis failed")
                        st.session_state.pop("analysis_result", None)
                        st.session_state.pop("analysis_input", None)
                        st.error(f"Analysis could not be completed ({type(exc).__name__}). Check the app logs for details.")
                    else:
                        st.session_state["analysis_result"] = analysis_result
                        st.session_state["analysis_input"] = list(user_lines)

        res = st.session_state.get("analysis_result")
        has_result = (
            res is not None
            and st.session_state.get("analysis_input") == user_lines
        )
        if res is not None and not has_result:
            st.info("The input has changed since the last analysis. Click **Run Derailment Risk Analysis** to refresh the results.")

        if has_result:
            st.markdown("#### Turn-by-Turn Moderator Chat Feed")
            for item in res["msg_details"]:
                st.markdown(
                    f'<div class="{item["card_class"]}">'
                    f'<b>Turn {item["index"]}:</b> {html.escape(item["text"])}<br/>'
                    f'<div class="meta-tag">State Tag: <code>{item["state"]}</code> | Toxicity: <b>{item["toxicity"]:.1%}</b> | Sentiment: <b>{item["sentiment"]:+.2f}</b></div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    with col_results:
        st.subheader("Moderator Intelligence Board")
        if has_result:
            m1, m2, m3 = st.columns(3)
            m1.metric("DistilBERT Risk Score", f"{res['risk_score']:.1%}")
            m2.metric("Matched Pattern ID", str(res['matched_pattern']['pattern_id']) if res['matched_pattern'] else "None")
            if res['matched_pattern']:
                ratio_label = f"{float(res['matched_pattern']['support_ratio']):.2f}"
                if not res['pattern_is_trustworthy']:
                    ratio_label += " (limited support)"
                m3.metric("Pattern Support Ratio", ratio_label)
            else:
                m3.metric("Pattern Support Ratio", "N/A")

            if "HIGH" in res["decision"]:
                st.error(f"**Triage signal:** {res['decision']}")
            elif "MODERATE" in res["decision"]:
                st.warning(f"**Triage signal:** {res['decision']}")
            else:
                st.info(f"**Triage signal:** {res['decision']}")

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
    st.markdown("Searchable catalog of state trajectories mined from the bundled processed conversations.")
    st.caption(
        f"Note: patterns with fewer than {MIN_HEALTHY_SUPPORT_FOR_TRUSTED_RATIO} healthy-case observations "
        f"have limited evidence; treat their support ratios as descriptive, not precise risk multipliers."
    )
    p_df = pd.DataFrame(mined_patterns)[['pattern_id', 'pattern_str', 'pattern_length', 'support_ratio', 'derailed_support', 'healthy_support']]
    p_df['Evidence strength'] = np.where(
        p_df['healthy_support'] >= MIN_HEALTHY_SUPPORT_FOR_TRUSTED_RATIO,
        "Stronger support",
        "Limited support",
    )
    search_pattern = st.text_input("Filter by pattern text or ID", placeholder="e.g. NEGATIVE or PAT_12")
    if search_pattern.strip():
        q = search_pattern.strip()
        mask = p_df["pattern_str"].str.contains(q, case=False, regex=False) | p_df["pattern_id"].astype(str).str.contains(q, case=False, regex=False)
        p_df = p_df.loc[mask]
    st.caption(f"Showing {len(p_df):,} pattern(s). Ratios are descriptive historical support ratios, not guaranteed causal effects.")
    st.dataframe(p_df, use_container_width=True, height=480, hide_index=True)

with tab3:
    st.markdown("### Database overview")
    st.markdown(
        '<div class="cg-panel"><div class="cg-panel-title">Document model</div>'
        '<div class="cg-panel-caption">A conversation is one document with an embedded utterances array. '
        'This keeps the thread and its message-level scores together for common moderator reads. '
        'Patterns and moderator notes live in separate collections.</div></div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    try:
        c1.metric("Conversation documents", f"{db['conversations'].count_documents({}):,}")
        c2.metric("Mined patterns", f"{db['mined_patterns'].count_documents({}):,}")
        c3.metric("Moderator notes", f"{db['moderator_notes'].count_documents({}):,}")
        c4.metric("MongoDB mode", "Persistent" if "persistent" in database_mode else "Demo")
        cache_hits = int(redis_client.get("convoguard:stats:toxicity_cache_hits") or 0)
        cache_misses = int(redis_client.get("convoguard:stats:toxicity_cache_misses") or 0)
        c5.metric("Redis cache H / M", f"{cache_hits:,} / {cache_misses:,}")
    except Exception:
        st.info("Database statistics are temporarily unavailable.")

    with st.expander("Schema and indexes", expanded=False):
        schema_col, index_col = st.columns(2)
        with schema_col:
            st.markdown("**Conversation document (example shape)**")
            example_doc = db["conversations"].find_one({})
            if example_doc:
                st.json({
                    "_id": example_doc.get("_id"),
                    "conversation_id": example_doc.get("conversation_id"),
                    "split": example_doc.get("split"),
                    "has_attack": example_doc.get("has_attack"),
                    "symbol_sequence": example_doc.get("symbol_sequence", [])[:8],
                    "utterances": example_doc.get("utterances", [])[:2],
                    "utterance_count": len(example_doc.get("utterances", [])),
                })
        with index_col:
            st.markdown("**Conversation collection indexes**")
            try:
                index_names = list(db["conversations"].index_information().keys())
                st.write(index_names)
            except Exception as exc:
                st.info(f"Index details unavailable ({type(exc).__name__}).")
            st.caption("Indexes support conversation-ID lookup, nested speaker/toxicity filters, and attack/split filtering.")

    st.markdown("---")
    st.markdown("### 1. Query embedded conversation documents")
    query_col, limit_col = st.columns([3, 1])
    with query_col:
        query_speaker = st.text_input(
            "Search by exact speaker name",
            value="Jytdog",
            help="This filters the nested utterances.speaker field.",
        )
    with limit_col:
        result_limit = st.selectbox("Maximum results", [1, 3, 5, 10], index=1)
    if query_speaker.strip():
        matching_docs = list(
            db["conversations"]
            .find({"utterances.speaker": query_speaker.strip()})
            .limit(int(result_limit))
        )
        st.caption(f"Query returned {len(matching_docs)} matching document(s).")
        if matching_docs:
            for document in matching_docs:
                with st.expander(
                    f"Conversation {document.get('conversation_id', document.get('_id'))} "
                    f"· {'Attack-labelled' if document.get('has_attack') else 'Healthy-labelled'}"
                ):
                    st.json(document)
        else:
            st.info(f"No conversation documents found for speaker \"{query_speaker.strip()}\".")
    else:
        st.caption("Enter a speaker name to search the embedded utterances array.")

    if st.button(
        "Explain speaker query plan",
        disabled=not bool(query_speaker.strip()),
        help="Shows index/execution statistics when connected to a real MongoDB server.",
    ):
        if "mongomock" in database_mode:
            st.info(
                "Server-side explain statistics require real MongoDB. "
                "Connect with MONGODB_URI to demonstrate the winning plan and index usage."
            )
        else:
            try:
                explain_result = db.command(
                    "explain",
                    {
                        "find": "conversations",
                        "filter": {"utterances.speaker": query_speaker.strip()},
                        "limit": int(result_limit),
                    },
                    verbosity="executionStats",
                )
                execution_stats = explain_result.get("executionStats", {})
                e1, e2, e3 = st.columns(3)
                e1.metric("Documents examined", f"{execution_stats.get('totalDocsExamined', 0):,}")
                e2.metric("Index keys examined", f"{execution_stats.get('totalKeysExamined', 0):,}")
                e3.metric("Returned", f"{execution_stats.get('nReturned', 0):,}")
                with st.expander("Winning query plan"):
                    st.json(explain_result.get("queryPlanner", {}).get("winningPlan", {}))
            except Exception as exc:
                logger.exception("MongoDB explain failed")
                st.warning(
                    f"MongoDB explain is unavailable ({type(exc).__name__}); "
                    "verify server version and permissions."
                )

    st.markdown("---")
    st.markdown("### 2. Moderator notes — Create, Read, Update, Delete")
    st.caption(
        "Notes are stored separately from the reference conversation corpus. "
        "Saving a note for an existing conversation creates it or updates the existing note."
    )
    write_col, read_col = st.columns(2, gap="large")
    with write_col:
        with st.form("moderator_note_form", clear_on_submit=False):
            note_conversation_id = st.text_input(
                "Conversation ID",
                placeholder="Paste a conversation_id from the search results",
            )
            note_text = st.text_area(
                "Moderator note",
                placeholder="Record why this thread needs follow-up...",
                height=120,
            )
            save_note = st.form_submit_button(
                "Save / update note", type="primary", use_container_width=True
            )
        if save_note:
            conv_id = note_conversation_id.strip()
            clean_note = note_text.strip()
            if not conv_id or not clean_note:
                st.warning("Enter both a conversation ID and a note.")
            elif not db["conversations"].find_one({"conversation_id": conv_id}, {"_id": 1}):
                st.warning("That conversation ID is not present in the loaded reference corpus.")
            else:
                now = datetime.now(timezone.utc).isoformat()
                db["moderator_notes"].update_one(
                    {"_id": conv_id},
                    {"$set": {
                        "conversation_id": conv_id,
                        "note": clean_note,
                        "updated_at": now,
                        "updated_by": "moderator-demo",
                    }},
                    upsert=True,
                )
                st.success("Moderator note saved.")
                st.rerun()

    with read_col:
        st.markdown("**Saved notes**")
        notes = list(
            db["moderator_notes"].find({}).sort("updated_at", -1).limit(50)
        )
        if notes:
            for note in notes:
                with st.expander(
                    f"{note.get('conversation_id', note.get('_id'))} · {note.get('updated_at', '')[:19]}"
                ):
                    st.write(note.get("note", ""))
                    st.caption(f"Updated by {note.get('updated_by', 'unknown')}")
                    if st.button(
                        "Delete this note",
                        key=f"delete-note-{note.get('_id')}",
                        type="secondary",
                    ):
                        db["moderator_notes"].delete_one({"_id": note["_id"]})
                        st.success("Note deleted.")
                        st.rerun()
        else:
            st.info("No moderator notes saved yet. Create one using the form.")

    st.markdown("---")
    st.markdown("### 3. Aggregation pipeline: most toxic-comment speakers")
    st.caption(
        "Uses $unwind to expand embedded utterances, $match to filter toxicity, "
        "$group to count by speaker, then $sort and $limit."
    )
    pipeline = [
        {"$unwind": "$utterances"},
        {"$match": {"utterances.toxicity": {"$gte": 0.3597}}},
        {"$group": {"_id": "$utterances.speaker", "toxic_comment_count": {"$sum": 1}}},
        {"$sort": {"toxic_comment_count": -1}},
        {"$limit": 5},
    ]
    try:
        aggregation_results = list(db["conversations"].aggregate(pipeline))
        top_speakers_df = pd.DataFrame(aggregation_results).rename(
            columns={"_id": "Speaker name", "toxic_comment_count": "Toxic comments"}
        )
        st.dataframe(top_speakers_df, use_container_width=True, hide_index=True)
        with st.expander("View the pipeline"):
            st.code(json.dumps(pipeline, indent=2), language="json")
    except Exception as exc:
        st.error(f"Aggregation failed ({type(exc).__name__}). Check MongoDB compatibility and app logs.")


with tab4:
    st.subheader("Sequential forecasting — measured results")
    st.markdown(
        "This page reads the benchmark output generated by the repository's reproducible "
        "conversation-level experiment. It does not fabricate or recompute scores in the UI."
    )
    benchmark_dir = PROJECT_DIR / "reports" / "generated"
    summary_path = benchmark_dir / "sequence_benchmark_summary.csv"
    report_path = benchmark_dir / "sequence_benchmark_metrics.json"

    if not summary_path.exists() or not report_path.exists():
        st.info(
            "Benchmark files are not present in this checkout yet. From the repository root, "
            "run the following command and then refresh this page:"
        )
        st.code(
            "python scripts/run_sequence_benchmark.py --output reports/generated",
            language="bash",
        )
    else:
        try:
            benchmark_summary = pd.read_csv(summary_path)
            benchmark_report = json.loads(report_path.read_text(encoding="utf-8"))
            required_columns = {
                "model", "prefix_label", "prefix_fraction", "macro_f1",
                "attack_f1", "attack_recall", "pr_auc", "roc_auc",
                "balanced_accuracy", "brier_score",
            }
            if missing := required_columns - set(benchmark_summary.columns):
                raise ValueError(
                    f"Benchmark CSV is missing columns: {sorted(missing)}"
                )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            st.error(
                f"Benchmark report could not be read ({type(exc).__name__}). "
                "Regenerate it using the command above."
            )
        else:
            integrity = benchmark_report.get("split_integrity", {})
            n1, n2, n3 = st.columns(3)
            n1.metric("Conversations", f"{benchmark_report.get('n_conversations', 0):,}")
            n2.metric("Training conversations", f"{benchmark_report.get('n_train', 0):,}")
            n3.metric("Holdout conversations", f"{benchmark_report.get('n_test', 0):,}")
            if integrity.get("train_holdout_id_overlap_count") == 0:
                st.success(
                    "Split audit passed: train/holdout conversation-ID overlap = 0. "
                    "These are internal holdout results, not official CGA-Wiki test results."
                )
            else:
                st.error(
                    "Split integrity audit did not report zero train/holdout overlap. "
                    "Do not use these metrics until the issue is resolved."
                )

            numeric_rows = benchmark_summary[
                benchmark_summary["prefix_fraction"].notna()
            ].copy()
            numeric_rows["prefix_fraction"] = pd.to_numeric(
                numeric_rows["prefix_fraction"], errors="coerce"
            )
            model_options = [
                model_name
                for model_name in [
                    "TF-IDF text baseline",
                    "Word + character TF-IDF",
                    "State-frequency ablation (orderless)",
                    "Ordered-state n-gram model",
                    "Hybrid text + state sequence",
                ]
                if model_name in set(numeric_rows["model"])
            ]
            selected_models = st.multiselect(
                "Models to compare",
                model_options,
                default=model_options,
                key="sequence_results_models",
            )
            metric_options = {
                "Macro-F1": "macro_f1",
                "Attack F1": "attack_f1",
                "Attack recall": "attack_recall",
                "PR-AUC": "pr_auc",
                "ROC-AUC": "roc_auc",
                "Balanced accuracy": "balanced_accuracy",
            }
            metric_label = st.selectbox(
                "Metric", list(metric_options), key="sequence_results_metric"
            )
            metric_column = metric_options[metric_label]
            chart_data = numeric_rows[
                numeric_rows["model"].isin(selected_models)
            ].pivot(
                index="prefix_label",
                columns="model",
                values=metric_column,
            )
            horizon_order = [label for label in ["25%", "50%", "75%", "100%"] if label in chart_data.index]
            chart_data = chart_data.reindex(horizon_order)
            if selected_models and not chart_data.empty:
                st.line_chart(chart_data)
            else:
                st.info("Select at least one model to draw the comparison.")

            st.caption(
                "The 25%, 50%, and 75% rows are the early-warning horizons. "
                "The 100% row uses the full conversation and is shown as a context-only comparison."
            )
            display_columns = [
                column for column in [
                    "model", "prefix_label", "n_train", "n_test", "accuracy",
                    "balanced_accuracy", "macro_f1", "attack_precision",
                    "attack_recall", "attack_f1", "pr_auc", "roc_auc",
                    "brier_score", "macro_f1_bootstrap_95ci",
                ] if column in benchmark_summary.columns
            ]
            st.dataframe(
                benchmark_summary[display_columns],
                use_container_width=True,
                hide_index=True,
            )
            with st.expander("Evaluation protocol and split audit"):
                st.write(benchmark_report.get("evaluation_protocol", ""))
                st.warning(benchmark_report.get("interpretation_limit", ""))
                st.json(
                    {
                        "random_seed": benchmark_report.get("random_seed"),
                        "positive_label": benchmark_report.get("positive_label"),
                        "split_integrity": integrity,
                        "model_descriptions": benchmark_report.get("model_descriptions", {}),
                    }
                )
