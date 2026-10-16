# ConversationGuardian

ConversationGuardian is a Streamlit research/demo application for exploring possible escalation in Wikipedia discussion threads. It combines NLP-based risk signals with mined state-sequence patterns and a document-database workbench.

> **Research/demo tool:** predictions are estimates for moderator review, not definitive findings or automatic enforcement decisions.

## What is included

- **Risk Simulator:** enter a multi-turn discussion or choose a preset. The app displays the DistilBERT conversation risk score, per-message Detoxify toxicity, VADER sentiment, state trajectory, and a supporting pattern match.
- **Pattern Catalog:** search mined sequence patterns and inspect their historical support counts. Small healthy-case counts are labelled as limited evidence; a support ratio is not proof of causation.
- **NoSQL Lab:** inspect the document schema, run nested-field conversation queries, view indexes, execute an aggregation pipeline, and create/read/update/delete moderator notes without modifying the reference conversation corpus.
- **Storage modes:** connect to real MongoDB using MONGODB_URI and a Redis cache using REDIS_URL, or run in clearly labelled in-memory mongomock/fakeredis demo modes.

## Architecture

    Streamlit UI (app/dashboard.py)
      ├── NLP signals: DistilBERT + Detoxify + VADER
      ├── Prefix/state pattern matching (data/patterns/mined_patterns.json)
      ├── Parquet data preparation (data/cleaned, data/features, data/sequences)
      └── Database layer (app/database.py)
           ├── conversations
           ├── mined_patterns
           ├── moderator_notes
           └── Redis cache (toxicity scores keyed by message hash, 24-hour TTL)

The dashboard loads its bundled cleaned training conversations, features, and state sequences. It does not download the raw Wikipedia discussion corpus at runtime.

## NoSQL document design

### conversations

One document represents one conversation. Its **utterances** array embeds the messages with the speaker, cleaned text, and toxicity score. Embedding keeps turn-level context together for the main thread-reading workflow.

Example shape (illustrative, shortened):

    {
      "_id": "conv_example-1",
      "conversation_id": "example-1",
      "split": "train",
      "has_attack": false,
      "symbol_sequence": ["NEUTRAL", "NEGATIVE", "POSITIVE"],
      "utterances": [
        {"speaker": "EditorA", "clean_text": "Please explain the source.", "toxicity": 0.02},
        {"speaker": "EditorB", "clean_text": "I disagree with that edit.", "toxicity": 0.04}
      ]
    }

### mined_patterns

One document per mined pattern, keyed by pattern ID and containing the state sequence, support counts, and historical support ratio.

### moderator_notes

A separate, mutable collection keyed by the conversation ID. The NoSQL Lab's CRUD demonstration is limited to this collection; it never edits or deletes reference conversation documents.

### Indexes

The app creates indexes for:
- unique conversation ID;
- nested utterances.speaker lookup;
- nested utterances.toxicity filtering;
- has_attack + split filtering;
- unique mined pattern ID;
- moderator-note conversation ID lookup.

These choices should be explained in terms of query patterns, not merely listed.

## Run locally

Use Python 3.10+ in a virtual environment.

Windows PowerShell:

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    streamlit run app/dashboard.py

The first run may download model files from Hugging Face and the Detoxify model. Keep internet access available for the initial run.

Optional local settings can be loaded from a .env file:

    Copy-Item .env.example .env

Edit the values only when using persistent services. Never commit .env; the repository ignores it. The default in-memory demo modes work without service credentials.

## Use a real MongoDB database

For an actual persistent database demonstration, configure a MongoDB Atlas or local MongoDB connection string in the environment. Do not commit credentials into this repository.

Windows PowerShell example:

    $env:MONGODB_URI = "mongodb+srv://<user>:<password>@<cluster>/<database>?retryWrites=true&w=majority"
    $env:REDIS_URL = "rediss://<user>:<password>@<host>:<port>"
    streamlit run app/dashboard.py

For Streamlit Cloud, add secrets named MONGODB_URI and REDIS_URL in the app's settings if using both services. Ensure the MongoDB and Redis network access rules allow the deployed app to connect.

The app verifies the connection using a ping. If a URI is configured but the connection fails, startup reports a failure instead of silently switching to a fake database.

The toxicity model scores are cached in Redis using a SHA-256 message hash and a 24-hour TTL. Message text itself is not written to the cache. Without REDIS_URL, the app uses in-memory fakeredis and labels that mode explicitly.

When MONGODB_URI is **not set**, the app uses mongomock in-memory storage and displays a warning. It is suitable for a quick UI demo, but it is not persistent and should not be presented as a live MongoDB deployment. When REDIS_URL is not set, fakeredis is used and its cache does not persist across restarts.

## Tests

Run the database layer regression tests:

    python -m unittest discover -s tests -v

They cover explicit demo-mode labelling, in-memory cache behavior, idempotent reference-data seeding, preservation of moderator notes, indexes, and validation of document IDs.

For a course-ready explanation of the MongoDB document schema, embedding decision, indexes, query plan, aggregation, CRUD, and Redis cache, see [docs/NOSQL_DESIGN.md](docs/NOSQL_DESIGN.md).

## Sequential modeling and experimental results

Run the conversation-level early-warning benchmark:

    python scripts/run_sequence_benchmark.py --output reports/generated

It compares a majority reference, word TF-IDF, word-plus-character TF-IDF, orderless state frequencies, ordered state n-grams, and a hybrid text-plus-state classifier at 25%, 50%, 75%, and 100% prefixes. Results include Macro-F1, attack precision/recall/F1, PR-AUC, ROC-AUC, balanced accuracy, Brier score, confidence intervals, confusion matrices, prediction probabilities, and an audit that train/holdout conversation IDs do not overlap.

The benchmark uses an internal holdout from the bundled processed training snapshot. It must not be described as official CGA-Wiki test performance. See [docs/RESULTS.md](docs/RESULTS.md) for the measured comparison table and interpretation, and [docs/ASSESSMENT_GUIDE.md](docs/ASSESSMENT_GUIDE.md) for the problem statement, evaluation protocol, presentation script, and viva questions. GitHub Actions also publishes full benchmark outputs as a downloadable artifact.

## Important limitations

- The configured risk thresholds are currently hand-set defaults and still need validation-set calibration before being described as statistically calibrated.
- Model outputs can be wrong and may misread context, sarcasm, policy debate, or quoted harmful language.
- mongomock is not a MongoDB server. A persistent database demonstration requires a working MONGODB_URI.
- This repository version is a prototype: the UI fix and database-layer tests should be exercised in the actual deployment environment before assessment.
