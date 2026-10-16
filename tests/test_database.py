import unittest

from app.database import connect_cache, connect_database, seed_reference_data


class DatabaseLayerTests(unittest.TestCase):
    def setUp(self):
        self.client, self.db, self.mode = connect_database()

    def test_without_uri_is_explicit_demo_mode(self):
        self.assertIn("mongomock", self.mode.lower())
        self.assertIn("not persistent", self.mode.lower())

    def test_demo_cache_supports_get_set_and_counters(self):
        cache, mode = connect_cache()
        self.assertIn("fakeredis", mode.lower())
        cache.set("test-key", "0.25", ex=60)
        self.assertEqual(cache.get("test-key"), "0.25")
        cache.incr("test-counter")
        self.assertEqual(int(cache.get("test-counter")), 1)

    def test_seeding_is_idempotent_and_preserves_moderator_notes(self):
        patterns = [
            {
                "pattern_id": "PAT_1",
                "pattern": ["NEGATIVE", "POSITIVE"],
                "pattern_str": "NEGATIVE -> POSITIVE",
                "support_ratio": 2.0,
                "healthy_support": 4,
            }
        ]
        conversations = [
            {
                "_id": "conv_c1",
                "conversation_id": "c1",
                "split": "train",
                "has_attack": False,
                "symbol_sequence": ["NEUTRAL"],
                "utterances": [
                    {
                        "speaker": "EditorA",
                        "clean_text": "Hello",
                        "toxicity": 0.01,
                    }
                ],
            }
        ]
        seed_reference_data(self.db, patterns, conversations)
        self.db["moderator_notes"].insert_one(
            {"_id": "c1", "conversation_id": "c1", "note": "Review later"}
        )

        # Reseeding should update stable reference documents, not drop collections.
        updated_conversation = dict(conversations[0])
        updated_conversation["utterances"] = [
            {
                "speaker": "EditorA",
                "clean_text": "Updated text",
                "toxicity": 0.02,
            }
        ]
        seed_reference_data(self.db, patterns, [updated_conversation])

        self.assertEqual(self.db["conversations"].count_documents({}), 1)
        self.assertEqual(self.db["mined_patterns"].count_documents({}), 1)
        saved = self.db["conversations"].find_one({"conversation_id": "c1"})
        self.assertEqual(saved["utterances"][0]["clean_text"], "Updated text")
        note = self.db["moderator_notes"].find_one({"_id": "c1"})
        self.assertEqual(note["note"], "Review later")
        indexes = self.db["conversations"].index_information()
        self.assertIn("speaker_lookup", indexes)
        self.assertIn("toxicity_lookup", indexes)

    def test_nested_query_aggregation_and_moderator_note_crud(self):
        conversations = [
            {
                "_id": "conv_c1",
                "conversation_id": "c1",
                "split": "train",
                "has_attack": False,
                "symbol_sequence": ["NEUTRAL", "NEGATIVE_TOXIC"],
                "utterances": [
                    {"speaker": "EditorA", "clean_text": "Hello", "toxicity": 0.02},
                    {"speaker": "EditorB", "clean_text": "Rude comment", "toxicity": 0.80},
                ],
            },
            {
                "_id": "conv_c2",
                "conversation_id": "c2",
                "split": "train",
                "has_attack": True,
                "symbol_sequence": ["NEGATIVE_TOXIC"],
                "utterances": [
                    {"speaker": "EditorB", "clean_text": "Another rude comment", "toxicity": 0.70},
                ],
            },
        ]
        seed_reference_data(self.db, [], conversations)

        matches = list(self.db["conversations"].find({"utterances.speaker": "EditorB"}))
        self.assertEqual({doc["conversation_id"] for doc in matches}, {"c1", "c2"})

        pipeline = [
            {"$unwind": "$utterances"},
            {"$match": {"utterances.toxicity": {"$gte": 0.5}}},
            {"$group": {"_id": "$utterances.speaker", "toxic_comment_count": {"$sum": 1}}},
            {"$sort": {"toxic_comment_count": -1}},
            {"$limit": 5},
        ]
        results = list(self.db["conversations"].aggregate(pipeline))
        self.assertEqual(results[0]["_id"], "EditorB")
        self.assertEqual(results[0]["toxic_comment_count"], 2)

        notes = self.db["moderator_notes"]
        notes.update_one(
            {"_id": "c1"},
            {"$set": {"conversation_id": "c1", "note": "Review context"}},
            upsert=True,
        )
        self.assertEqual(notes.find_one({"_id": "c1"})["note"], "Review context")
        notes.update_one({"_id": "c1"}, {"$set": {"note": "Reviewed"}})
        self.assertEqual(notes.find_one({"_id": "c1"})["note"], "Reviewed")
        notes.delete_one({"_id": "c1"})
        self.assertIsNone(notes.find_one({"_id": "c1"}))

    def test_rejects_conversation_without_document_id(self):
        with self.assertRaisesRegex(ValueError, "_id"):
            seed_reference_data(
                self.db,
                [{"pattern_id": "PAT_1", "pattern": ["NEUTRAL"]}],
                [{"conversation_id": "missing-id"}],
            )


if __name__ == "__main__":
    unittest.main()
