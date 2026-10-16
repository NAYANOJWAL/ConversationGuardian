# ConversationGuardian

ConversationGuardian is a Streamlit research/demo application for exploring possible escalation in Wikipedia discussion threads. It combines NLP-based risk signals with mined state-sequence patterns and a document-database workbench.

> **Research/demo tool:** predictions are estimates for moderator review, not definitive findings or automatic enforcement decisions.

## What is included

- **Risk Simulator:** enter a multi-turn discussion or choose a preset. The app displays the DistilBERT conversation risk score, per-message Detoxify toxicity, VADER sentiment, state trajectory, and a supporting pattern match.
- **Pattern Catalog:** search mined sequence patterns and inspect their historical support counts. Small healthy-case counts are labelled as limited evidence; a support ratio is not proof of causation.
- **NoSQL Lab:** inspect the document schema, run nested-field conversation queries, view indexes, execute an aggregation pipeline, and create/read/update/delete moderator notes without modifying the reference conversation corpus.
- **Storage modes:** connect to a real MongoDB database using the MONGODB_URI environment setting, or run in clearly labelled in-memory mongomock demo mode.

## Architecture

    Streamlit UI (app/dashboard.py)
      ├── NLP signals: DistilBERT + Detoxify + VADER
      ├── Prefix/state pattern matching (data/patterns/mined_patterns.json)
      ├── Parquet data preparation (data/cleaned, data/features, data/sequences)
      └── Database layer (app/database.py)
           ├── conversations
           ├── mined_patterns
           └── moderator_notes

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

## Use a real MongoDB database

For an actual persistent database demonstration, configure a MongoDB Atlas or local MongoDB connection string in the environment. Do not commit credentials into this repository.

Windows PowerShell example:

    $env:MONGODB_URI = "mongodb+srv://<user>:<password>@<cluster>/<database>?retryWrites=true&w=majority"
    streamlit run app/dashboard.py

For Streamlit Cloud, add a secret named MONGODB_URI in the app's settings. Ensure the Atlas network access rules allow the deployed app to connect.

The app verifies the connection using a ping. If a URI is configured but the connection fails, startup reports a failure instead of silently switching to a fake database.

When MONGODB_URI is **not set**, the app uses mongomock in-memory storage and displays a warning. It is suitable for a quick UI demo, but it is not persistent and should not be presented as a live MongoDB deployment.

## Tests

Run the database layer regression tests:

    python -m unittest discover -s tests -v

They cover explicit demo-mode labelling, idempotent reference-data seeding, preservation of moderator notes, indexes, and validation of document IDs.

## Important limitations

- The configured risk thresholds are currently hand-set defaults and still need validation-set calibration before being described as statistically calibrated.
- Model outputs can be wrong and may misread context, sarcasm, policy debate, or quoted harmful language.
- mongomock is not a MongoDB server. A persistent database demonstration requires a working MONGODB_URI.
- This repository version is a prototype: the UI fix and database-layer tests should be exercised in the actual deployment environment before assessment.
