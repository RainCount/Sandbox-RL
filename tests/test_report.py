from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from swe_rl.report import extract_reward_series


class ReportTests(unittest.TestCase):
    def test_extract_reward(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            path.write_text(
                '{"global_step":1,"critic/rewards/mean":0.1}\n{"global_step":2,"critic/rewards/mean":0.2}\n',
                encoding="utf-8",
            )
            steps, values, key = extract_reward_series(path)
            self.assertEqual(steps, [1, 2])
            self.assertEqual(values, [0.1, 0.2])
            self.assertEqual(key, "critic/rewards/mean")

    def test_extract_reward_from_verl_file_logger_envelope(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            path.write_text(
                '{"step":10,"data":{"critic/rewards/mean":0.25,"training/global_step":10}}\n',
                encoding="utf-8",
            )

            steps, values, key = extract_reward_series(path)

            self.assertEqual(steps, [10])
            self.assertEqual(values, [0.25])
            self.assertEqual(key, "critic/rewards/mean")


if __name__ == "__main__":
    unittest.main()
