import unittest

import numpy as np
import pandas as pd

from scripts.run_sequence_benchmark import (
    normalize_sequence,
    run_benchmark,
    take_prefix,
)


class SequenceBenchmarkTests(unittest.TestCase):
    def test_normalize_sequence_accepts_json_and_lists(self):
        self.assertEqual(
            normalize_sequence('["NEUTRAL", "NEGATIVE"]'),
            ["NEUTRAL", "NEGATIVE"],
        )
        self.assertEqual(
            normalize_sequence(np.asarray(["POSITIVE", "NEUTRAL"])),
            ["POSITIVE", "NEUTRAL"],
        )

    def test_prefix_preserves_order_and_excludes_future_events(self):
        events = ["TURN_1", "TURN_2", "TURN_3", "TURN_4"]
        self.assertEqual(take_prefix(events, 0.25), ["TURN_1"])
        self.assertEqual(take_prefix(events, 0.50), ["TURN_1", "TURN_2"])
        self.assertEqual(take_prefix(events, 0.75), ["TURN_1", "TURN_2", "TURN_3"])
        self.assertEqual(take_prefix(events, 1.00), events)

    def test_benchmark_splits_at_conversation_level_and_reports_all_horizons(self):
        records = []
        for i in range(40):
            label = int(i % 2)
            records.append(
                {
                    "conversation_id": f"conv-{i}",
                    "label": label,
                    "messages": [
                        "policy source consensus" if not label else "insult attack conflict",
                        "neutral edit discussion" if not label else "rude toxic dispute",
                        "thanks for discussion" if not label else "you are abusive",
                        f"turn repeat token{i % 4}",
                    ],
                    "states": [
                        "NEUTRAL",
                        "NEGATIVE_TOXIC" if label else "POSITIVE",
                        "NEGATIVE" if label else "NEUTRAL",
                        f"TURN_{i % 3}",
                    ],
                    "message_count": 4,
                    "state_count": 4,
                    "source_split": "train",
                }
            )
        report, predictions = run_benchmark(pd.DataFrame(records))
        self.assertEqual(report["n_conversations"], 40)
        self.assertEqual(report["n_train"] + report["n_test"], 40)
        self.assertEqual(report["split_integrity"]["train_holdout_id_overlap_count"], 0)
        self.assertEqual(len(report["metrics"]), 21)
        self.assertEqual(len(predictions), report["n_test"])
        self.assertTrue(all(row["prefix_label"] for row in report["metrics"]))
        model_names = {row["model"] for row in report["metrics"]}
        self.assertIn("TF-IDF text baseline", model_names)
        self.assertIn("Word + character TF-IDF", model_names)
        self.assertIn("State-frequency ablation (orderless)", model_names)
        self.assertIn("Ordered-state n-gram model", model_names)
        self.assertIn("Hybrid text + state sequence", model_names)
        self.assertIn("not the official benchmark test set", report["interpretation_limit"])


if __name__ == "__main__":
    unittest.main()
