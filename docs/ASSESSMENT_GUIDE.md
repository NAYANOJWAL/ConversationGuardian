# Assessment Guide: NoSQL + Sequential Modeling

This guide is written for a faculty demonstration. It separates what the application demonstrates from what has actually been experimentally measured.

## 1. Problem statement

Wikipedia talk pages contain multi-turn discussions about edits, sources, neutrality, and policy. A conversation can remain productive, become tense, or eventually contain a personal attack. Moderators benefit from two kinds of evidence: (1) a sequential model that estimates whether an attack-labelled conversation can be identified from an early chronological prefix, and (2) a document database that stores discussion context and supports targeted moderator queries.

**Project question:** How useful are the initial 25%, 50%, and 75% of a conversation for predicting whether the conversation is attack-labelled, and how can a document-oriented database support inspection and moderation workflows?

**Scope:** This prototype focuses on offline processed Wikipedia data, exploratory early-warning experiments, and an interactive database demonstration. It is not a production enforcement system and does not guarantee that future attacks can be predicted from every conversation.

## 2. Deliverables for the two courses

### Sequential modeling

The reproducible benchmark in scripts/run_sequence_benchmark.py:
- builds one row per conversation_id;
- holds out conversations, never individual comments, to prevent comments from the same thread crossing the train/holdout boundary;
- trains only on the training portion of this internal split;
- reports metrics at 25%, 50%, 75%, and 100% prefix lengths;
- compares a majority-class reference, a word TF-IDF baseline, a word-plus-character TF-IDF baseline, an orderless state-frequency ablation, an ordered state-token n-gram classifier, and a hybrid text-plus-state model;
- fixes the decision threshold at 0.5 rather than tuning on the holdout; records hashes of the train/holdout ID sets and asserts zero overlap;
- reports macro-F1, attack precision/recall/F1, PR-AUC, ROC-AUC, balanced accuracy, Brier score, confusion matrix, and a bootstrap interval for macro-F1.

Run it from the repository root:

    python scripts/run_sequence_benchmark.py --output reports/generated

Outputs:
- reports/generated/sequence_benchmark_metrics.json: protocol, split sizes, model metrics, confusion matrices and interpretation notes;
- reports/generated/sequence_benchmark_summary.csv: compact comparison table;
- reports/generated/sequence_benchmark_predictions.csv: one row per holdout conversation with predicted probabilities.

The GitHub Actions workflow runs this benchmark and publishes these files as an artifact. Download the artifact from the successful workflow run if the reports are not present locally.

**Interpretation caution:** the benchmark is an internal holdout made from the repository's bundled processed training snapshot. It is not the official CGA-Wiki benchmark test result and must not be presented as one. The full-prefix results may be easier because the full conversation can include the attack itself; the early-prefix results are more relevant to the early-warning question. Report all horizons, not only the best score. No metric should be claimed until the corresponding workflow run passes and the generated report has been inspected.

### NoSQL

The NoSQL Lab demonstrates:
- a conversation document with an embedded utterances array, keeping the thread and per-message signals together;
- a separate mined_patterns collection for reusable pattern metadata;
- a separate moderator_notes collection for mutable moderation notes, so the reference corpus is not altered;
- nested-field queries, such as utterances.speaker;
- a compound-field query on has_attack + split, illustrating how a filter maps to an index;
- a multi-stage aggregation pipeline using $unwind, $match, $group, $sort, and $limit;
- indexes matched to query patterns;
- optional Redis cache for toxicity scores, with a SHA-256 key and 24-hour TTL.

The database layer supports persistent MongoDB through MONGODB_URI and persistent Redis through REDIS_URL. Without these variables, the app deliberately uses mongomock and fakeredis in-memory substitutes. **In-memory mode must not be described to faculty as persistent MongoDB or Redis.**

To configure persistent services, set environment variables before launch:

Windows PowerShell:

    $env:MONGODB_URI = "<your MongoDB connection string>"
    $env:REDIS_URL = "<your Redis connection string>"
    streamlit run app/dashboard.py

Do not commit credentials or put passwords in screenshots. A real deployment must be tested after connection setup because CI does not have your credentials or external database service.

## 3. Recommended live demo (6–8 minutes)

1. **Problem (45 seconds):** explain that moderators need early warning, not automatic punishment. The model provides a risk estimate for review.
2. **Sequential result (90 seconds):** show the generated metrics table at 25/50/75/100%. Compare the hybrid model with the ordered-state and orderless ablation, text baseline, and majority baseline. Explain macro-F1, attack recall, PR-AUC, and the fixed 0.5 threshold. Do not cherry-pick.
3. **Risk Simulator (90 seconds):** select a healthy preset, run it, then select an escalation preset and run it. Show that per-message toxicity/sentiment, conversation risk, and sequence-pattern evidence are separate signals. Note that the output is a recommendation for moderator review.
4. **Document schema (60 seconds):** open NoSQL Lab, show one conversation document and explain why utterances are embedded.
5. **Query and aggregation (60 seconds):** search for a known speaker, then explain every stage of the $unwind aggregation by reading its input/output shape.
6. **CRUD and persistence (60 seconds):** save a moderator note, read it, update it, and delete it. If claiming persistence, restart or refresh the app and prove that the note remains with real MongoDB configured.
7. **Close (30 seconds):** state current limitations and one next step.

Always run the demo once before assessment. Keep a backup set of screenshots and the successful benchmark report, but do not imply screenshots prove external-service persistence.

## 4. What to explain in viva

- Why is conversation_id the split unit rather than a comment row?
- What exactly does the label has_attack mean?
- What information is available at the 25% prefix, and what later information is intentionally excluded?
- Why is macro-F1 included when class proportions may be imbalanced?
- What does PR-AUC tell us that accuracy does not?
- Why embed utterances instead of creating a separate collection for every message?
- Which queries justify the indexes? What write/storage cost do indexes add?
- Why are moderator notes separate from source conversations?
- What is the practical difference between MongoDB and mongomock? Between Redis and fakeredis?
- Why is a toxicity model score not equivalent to a confirmed personal attack?
- What experiment would be needed before claiming reliable generalization to unseen Wikipedia conversations?

## 5. Assessment-alignment checklist

### Problem statement
- [ ] Problem, target label, intended user, and decision supported by the tool are explained clearly.
- [ ] Dataset source and known limitations are stated.
- [ ] "Early warning" is tied to the prefix horizons, not just full-conversation classification.

### Results
- [ ] Attach the successful benchmark JSON/CSV from the current branch.
- [ ] Present every model/horizon row and the majority baseline, including the hybrid model.
- [ ] Explain false negatives as well as false positives.
- [ ] Do not report the official test set as freshly evaluated or claim the held-out history is untouched; this repository's earlier workflow history included access to test metrics.
- [ ] If BERT/DistilBERT metrics are shown, identify the exact model checkpoint, data split, threshold, and generation script. Do not mix its scores with the internal TF-IDF/state-sequence holdout table.

### Demo
- [ ] Run locally or deployed in advance with model artifacts cached.
- [ ] Use real MongoDB/Redis only when configured and verified; otherwise disclose in-memory demo mode.
- [ ] Show successful query, aggregation, and note CRUD.
- [ ] Have a fallback screenshot and downloaded report, clearly labelled as backup evidence.

### NoSQL design
- [ ] Explain embedding vs referencing choices.
- [ ] Explain every index in terms of a real query.
- [ ] Demonstrate CRUD and aggregation with visible output.
- [ ] Explain how updates avoid deleting the loaded reference corpus.

## 6. Claims to avoid

Do not claim that the model "prevents" attacks, that a support ratio proves causation, that a mock database is persistent, or that the experiment generalizes to all Wikipedia discussions. Prefer "risk estimate", "attack-labelled conversation", "internal conversation-level holdout", and "evidence for moderator review".
