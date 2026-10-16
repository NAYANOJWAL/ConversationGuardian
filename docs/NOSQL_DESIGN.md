# NoSQL Design Note

## Problem addressed by the database layer

A moderator needs to inspect a discussion thread and its message-level signals, search for threads involving a speaker, see recurring escalation trajectories, and record moderation follow-up without changing the reference dataset. ConversationGuardian models this workload in MongoDB and uses Redis as an optional cache for repeated toxicity inference.

## Collections and document shape

### 1. conversations — read-heavy reference corpus

One document represents one conversation. Message turns are embedded in the utterances array.

Illustrative shape:

    {
      "_id": "conv_example-1",
      "conversation_id": "example-1",
      "split": "train",
      "has_attack": false,
      "symbol_sequence": ["NEUTRAL", "NEGATIVE", "POSITIVE"],
      "utterances": [
        {
          "speaker": "EditorA",
          "clean_text": "Please explain the source.",
          "toxicity": 0.02
        },
        {
          "speaker": "EditorB",
          "clean_text": "I disagree with that edit.",
          "toxicity": 0.04
        }
      ]
    }

**Why embed utterances?** The main UI retrieves a thread together with its turns. Embedding avoids a second query for every thread and keeps speaker, text, and toxicity in the same parent document. It is appropriate for the current read pattern, but unusually large conversations could create large documents; a production version should check MongoDB's document-size constraints and consider a separate utterances collection if threads grow too large.

### 2. mined_patterns — analytical reference data

One document per mined pattern. Documents keep the pattern ID, state-token sequence, descriptive support counts, support ratio, and representative conversation IDs. This collection is read by the Pattern Catalog and risk simulator.

Pattern support ratios are descriptive historical statistics. They are not causal risk multipliers and do not multiply the DistilBERT score.

### 3. moderator_notes — mutable application data

A separate collection stores moderator notes, keyed by conversation ID. One note per conversation is supported by using the conversation ID as the note document's _id. A note can be created, read, updated, or deleted without changing the reference conversations or pattern data. This is the collection used for the CRUD demonstration.

## Index design

Indexes are created by app/database.py based on actual queries in the dashboard.

| Index | Query/workload it supports | Reason |
|---|---|---|
| Unique conversation_id | Query by conversation ID; prevent duplicate logical conversations | Enforces identity at the application level |
| utterances.speaker | Exact speaker match inside the embedded utterances array | Supports nested-array speaker search |
| utterances.toxicity | Filter on high-toxicity message scores | Supports toxicity-based analyses |
| has_attack + split | Filter conversations by label and data split | Supports label/split analytics |
| Unique pattern_id | Look up a mined pattern by its ID | Enforces stable pattern identity |
| moderator_notes.conversation_id | Find notes for a conversation | Supports note retrieval by thread |

Indexes speed up matching queries but add storage and write overhead. The NoSQL Lab displays index names. With real MongoDB configured, the “Explain speaker query plan” action displays execution statistics and the winning plan. In mongomock mode, server-side explain statistics are intentionally unavailable.

## Representative query patterns

### Find conversations containing a particular speaker

    db.conversations.find(
        {"utterances.speaker": "Jytdog"}
    ).limit(3)

This is a nested-field predicate over the embedded utterances array.

### Count high-toxicity comments by speaker

    [
      {"$unwind": "$utterances"},
      {"$match": {"utterances.toxicity": {"$gte": 0.3597}}},
      {
        "$group": {
          "_id": "$utterances.speaker",
          "toxic_comment_count": {"$sum": 1}
        }
      },
      {"$sort": {"toxic_comment_count": -1}},
      {"$limit": 5}
    ]

- $unwind expands the embedded utterances so each comment is an aggregation row.
- $match keeps comments at or above the displayed threshold.
- $group computes the count per speaker.
- $sort ranks speakers by count.
- $limit keeps the top five results.

These counts are based on the chosen Detoxify threshold and are not equivalent to confirmed personal attacks.

## Seed safety and persistence

Reference data is inserted using idempotent upserts. The app does not drop collections during startup, so notes are not wiped when reference data is reseeded. Conversation and pattern indexes are created when seeding is complete.

When MONGODB_URI is set, app/database.py pings the configured server and selects the database specified in the URI, defaulting to ConversationGuardian when no database name is provided. If connection fails, startup reports a failure instead of silently changing to fake storage.

When no MONGODB_URI is set, the app uses mongomock. This makes the app easier to run locally but data is only in memory and does not persist after restart. It must be disclosed as a simulated database in a presentation.

## Redis cache

The app caches per-message toxicity model scores by a SHA-256 hash of the text and a 24-hour TTL. Raw message text is not used as the cache key and is not written into Redis by the cache helper. With REDIS_URL configured, scores use persistent Redis; without it, the app uses an explicitly labelled in-memory fakeredis substitute.

The cache reduces repeated model inference when identical messages are submitted again. Cache hit/miss counters are displayed in the NoSQL Lab. They are demonstrations, not a formal latency benchmark.

## Validation and remaining limitations

The repository's automated tests cover in-memory mode labelling, cache get/set/counters, idempotent reference-data upserts, preservation of moderator notes, index creation, and validation of document IDs. GitHub Actions also compiles the Python code.

These tests do not prove a live MongoDB or Redis deployment works. Before presenting persistence, configure MONGODB_URI (and optionally REDIS_URL), launch the app, create a note, refresh/restart the app, and verify that the note remains. The application currently has no user authentication or authorization and should be presented as a course prototype, not a production moderation platform.
