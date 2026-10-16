"""MongoDB connection and seed helpers for ConversationGuardian.

A real MongoDB URI enables persistent storage. Without one, the app uses an
explicitly labelled mongomock database for local/demo use.
"""
from __future__ import annotations

from typing import Any, Iterable

import mongomock
from pymongo import ASCENDING, MongoClient, ReplaceOne
from pymongo.database import Database


def connect_database(mongodb_uri: str | None = None) -> tuple[Any, Database, str]:
    """Return (client, database, mode_label). Never silently mask bad credentials."""
    if mongodb_uri and mongodb_uri.strip():
        client = MongoClient(mongodb_uri.strip(), serverSelectionTimeoutMS=5000)
        # Fail early and clearly if the provided connection string is not usable.
        client.admin.command("ping")
        return client, client["ConversationGuardian"], "MongoDB connected (persistent)"

    client = mongomock.MongoClient()
    return client, client["ConversationGuardian"], "Demo mode: mongomock (in-memory, not persistent)"


def _replace_operations(documents: Iterable[dict[str, Any]]) -> list[ReplaceOne]:
    operations: list[ReplaceOne] = []
    for document in documents:
        doc = dict(document)
        if "_id" not in doc:
            raise ValueError("Every MongoDB document must have an _id before seeding.")
        operations.append(ReplaceOne({"_id": doc["_id"]}, doc, upsert=True))
    return operations


def seed_reference_data(
    db: Database,
    mined_patterns: Iterable[dict[str, Any]],
    conversations: Iterable[dict[str, Any]],
) -> None:
    """Idempotently seed reference collections without dropping user data."""
    pattern_docs = []
    for pattern in mined_patterns:
        doc = dict(pattern)
        doc["_id"] = str(doc.get("pattern_id", doc.get("_id", "")))
        if doc["_id"] == "":
            raise ValueError("Mined patterns must contain pattern_id or _id.")
        pattern_docs.append(doc)

    conversation_docs = [dict(doc) for doc in conversations]
    for collection_name, docs in (
        ("mined_patterns", pattern_docs),
        ("conversations", conversation_docs),
    ):
        if isinstance(db.client, mongomock.MongoClient):
            # mongomock 4.3 is not compatible with every new PyMongo bulk-replace
            # option yet, so use its supported single-document upsert API.
            for doc in docs:
                if "_id" not in doc:
                    raise ValueError(
                        "Every MongoDB document must have an _id before seeding."
                    )
                db[collection_name].replace_one(
                    {"_id": doc["_id"]}, doc, upsert=True
                )
        else:
            operations = _replace_operations(docs)
            if operations:
                db[collection_name].bulk_write(operations, ordered=False)

    # Indexes make the dashboard's common queries explicit and demonstrable.
    db["conversations"].create_index(
        [("conversation_id", ASCENDING)], unique=True, name="conversation_id_unique"
    )
    db["conversations"].create_index(
        [("utterances.speaker", ASCENDING)], name="speaker_lookup"
    )
    db["conversations"].create_index(
        [("utterances.toxicity", ASCENDING)], name="toxicity_lookup"
    )
    db["conversations"].create_index(
        [("has_attack", ASCENDING), ("split", ASCENDING)], name="attack_split_lookup"
    )
    db["mined_patterns"].create_index(
        [("pattern_id", ASCENDING)], unique=True, name="pattern_id_unique"
    )
    db["moderator_notes"].create_index(
        [("conversation_id", ASCENDING)], name="note_conversation_lookup"
    )
