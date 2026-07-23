from __future__ import annotations

import unittest

from swe_rl.agent.trajectory import TokenTrajectory, TrajectoryAlignmentError, transition_suffix


class TokenTrajectoryTests(unittest.TestCase):
    def test_model_and_observation_tokens_have_distinct_masks(self):
        trajectory = TokenTrajectory([1, 2])
        trajectory.append_generation([3, 4], [-0.1, -0.2])

        complete = trajectory.append_observation([5, 6])

        self.assertTrue(complete)
        self.assertEqual(trajectory.response_ids, [3, 4, 5, 6])
        self.assertEqual(trajectory.response_mask, [1, 1, 0, 0])
        self.assertEqual(trajectory.response_logprobs, [-0.1, -0.2, 0.0, 0.0])

    def test_observation_limit_is_reported(self):
        trajectory = TokenTrajectory([1])
        trajectory.append_generation([2], None)

        complete = trajectory.append_observation([3, 4], limit=1)

        self.assertFalse(complete)
        self.assertEqual(trajectory.response_ids, [2, 3])

    def test_transition_suffix_removes_a_generated_stop_token(self):
        self.assertEqual(
            transition_suffix([10, 20, 30, 40, 50], [20, 30], [7, 40]),
            [50],
        )

    def test_transition_suffix_requires_one_marker(self):
        with self.assertRaises(TrajectoryAlignmentError):
            transition_suffix([1, 2, 1, 2], [1, 2], [])

if __name__ == "__main__":
    unittest.main()
