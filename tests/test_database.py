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

    def test_rejects_conversation_without_document_id(self):
        with self.assertRaisesRegex(ValueError, "_id"):
            seed_reference_data(
                self.db,
                [{"pattern_id": "PAT_1", "pattern": ["NEUTRAL"]}],
                [{"conversation_id": "missing-id"}],
            )


if __name__ == "__main__":
    unittest.main()
