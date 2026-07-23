from __future__ import annotations

import unittest

from swe_rl.schema import RewardResult, TraceEpisode, TraceStep
from swe_rl.storage.layout import RunLayout


class ContractTests(unittest.TestCase):
    def test_trace_contract(self):
        steps = [
            TraceStep(step=1, action="ls", observation="x"),
            TraceStep(step=2, action="edit", observation="ok"),
            TraceStep(step=3, action="submit", observation="ok", reward=1.0, done=True),
        ]
        reward = RewardResult(1.0, True, 1, 1, 2, 2, 0, True)
        episode = TraceEpisode("x", "model", 0, steps, reward, "patch", [], 10, 4, "run", 0)
        episode.validate()
        self.assertIn('"schema_version":"2.0"', episode.to_json())
        self.assertIn('"model_response":""', episode.to_json())

    def test_layout_is_round_scoped(self):
        layout = RunLayout("project", "run-a", 2)
        self.assertEqual(layout.train_prompts, "project/runs/run-a/rounds/002/datasets/train.parquet")
        self.assertTrue(layout.marker("trained").endswith("status/trained.json"))


if __name__ == "__main__":
    unittest.main()
