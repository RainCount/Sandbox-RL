from __future__ import annotations

import unittest

from swe_rl.agent.policy import looks_like_environment_mutation, looks_like_focused_test


class AgentPolicyTests(unittest.TestCase):
    def test_recognizes_scratch_reproducer_as_focused_test(self):
        self.assertTrue(looks_like_focused_test("cd /testbed && python test_fix.py"))
        self.assertTrue(looks_like_focused_test("python /tmp/reproduce_bug.py"))
        self.assertFalse(looks_like_focused_test("python -c 'print(1)'"))

    def test_rejects_package_install_variants(self):
        self.assertTrue(looks_like_environment_mutation("apt-get install -y appdirs"))
        self.assertTrue(looks_like_environment_mutation("python -m pip install appdirs"))
        self.assertFalse(looks_like_environment_mutation("python -m pytest tests/test_x.py"))


if __name__ == "__main__":
    unittest.main()
