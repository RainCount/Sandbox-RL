from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from swe_rl.data.filter_instances import filter_parquet

from swe_rl.data.prepare import repository_stratified_select, split_train_eval


def _rows():
    return (
        [{"repo": "a/repo", "instance_id": f"a-{index}"} for index in range(10)]
        + [{"repo": "b/repo", "instance_id": f"b-{index}"} for index in range(10)]
        + [{"repo": "c/repo", "instance_id": f"c-{index}"} for index in range(10)]
    )


class DataTests(unittest.TestCase):
    def test_selection_is_deterministic_and_balanced(self):
        first = repository_stratified_select(_rows(), 9, 42)
        second = repository_stratified_select(_rows(), 9, 42)
        self.assertEqual(first, second)
        counts = {
            repo: sum(item["repo"] == repo for item in first) for repo in {item["repo"] for item in first}
        }
        self.assertEqual(set(counts.values()), {3})

    def test_train_eval_do_not_overlap(self):
        train, evaluation = split_train_eval(_rows(), eval_size=9, seed=42)
        self.assertFalse(
            {item["instance_id"] for item in train} & {item["instance_id"] for item in evaluation}
        )

    def test_filters_excluded_instance_from_staged_parquet(self):
        rows = [
            {"extra_info": {"instance_id": "keep"}, "value": 1},
            {"extra_info": {"instance_id": "bad"}, "value": 2},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.parquet"
            pq.write_table(pa.Table.from_pylist(rows), path)

            result = filter_parquet(path, {"bad"})
            filtered = pq.read_table(path).to_pylist()

        self.assertEqual(result["removed"], ["bad"])
        self.assertEqual([row["extra_info"]["instance_id"] for row in filtered], ["keep"])


if __name__ == "__main__":
    unittest.main()
