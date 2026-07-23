from __future__ import annotations

import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from swe_rl.config import TCRConfig
from swe_rl.sandbox.ags import (
    AGSSWEEnvironment,
    CommandResult,
    _truncate_observation,
    _is_top_level_test_path,
    _validate_editor_path,
)


class ObservationTests(unittest.TestCase):
    def test_truncates_lines_before_characters(self):
        output = "".join(f"line-{index:03d}\n" for index in range(200))

        truncated = _truncate_observation(output, maximum=10_000, maximum_lines=10)

        self.assertIn("190 lines omitted", truncated)
        self.assertIn("line-000", truncated)
        self.assertIn("line-199", truncated)
        self.assertNotIn("line-100", truncated)

    def test_truncates_single_long_line_from_both_ends(self):
        truncated = _truncate_observation("a" * 100 + "b" * 100, maximum=40)

        self.assertTrue(truncated.startswith("a" * 20))
        self.assertTrue(truncated.endswith("b" * 20))
        self.assertIn("160 characters omitted", truncated)


class EditorTests(unittest.TestCase):
    def _environment(self) -> AGSSWEEnvironment:
        environment = object.__new__(AGSSWEEnvironment)
        environment.execute = Mock(return_value=CommandResult("ok", 0, 0.1))
        return environment

    def test_view_uses_numbered_focused_range(self):
        environment = self._environment()

        result = environment._view(
            'STR_VIEW {"path":"/testbed/pkg/x.py","view_range":[10,20]}', 30
        )

        self.assertEqual(result.return_code, 0)
        command = environment.execute.call_args.args[0]
        self.assertIn("nl -ba -- /testbed/pkg/x.py", command)
        self.assertIn("10,20p", command)
        environment.execute.assert_called_once_with(command, timeout=30, intercept=False)

    def test_editor_rejects_escape_and_top_level_test_edit(self):
        self.assertIsNotNone(_validate_editor_path("/testbed/../etc/passwd"))
        environment = self._environment()
        command = (
            "STR_REPLACE /testbed/tests/test_bug.py\n"
            "<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE"
        )

        result = environment._replace(command, 30)

        self.assertEqual(result.return_code, 2)
        self.assertIn("TEST_EDIT_REFUSED", result.output)
        environment.execute.assert_not_called()

    def test_root_level_scratch_helpers_are_test_paths(self):
        self.assertTrue(_is_top_level_test_path("/testbed/test_fix.py"))
        self.assertTrue(_is_top_level_test_path("/testbed/reproduce_issue.py"))
        self.assertTrue(_is_top_level_test_path("/testbed/tests/unit/test_bug.py"))
        self.assertFalse(_is_top_level_test_path("/testbed/src/check_framework.py"))

    def test_replace_script_reads_and_writes_utf8(self):
        environment = self._environment()
        command = (
            "STR_REPLACE /testbed/pkg/x.py\n"
            "<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE"
        )

        environment._replace(command, 30)

        encoded = environment.execute.call_args.args[0].split()[1]
        import base64

        script = base64.b64decode(encoded).decode()
        self.assertIn("read_text(encoding='utf-8')", script)
        self.assertIn("write_text(y,encoding='utf-8')", script)

    def test_create_marks_new_production_file_for_git_diff(self):
        environment = self._environment()

        result = environment._create(
            'STR_CREATE {"path":"/testbed/pkg/new.py","file_text":"value = 1\\n"}', 30
        )

        self.assertEqual(result.return_code, 0)
        command = environment.execute.call_args.args[0]
        self.assertIn("git add -N -- /testbed/pkg/new.py", command)
        environment.execute.assert_called_once_with(
            command, timeout=30, truncate=False, intercept=False
        )


class TCRFallbackTests(unittest.TestCase):
    def _environment(self) -> AGSSWEEnvironment:
        environment = object.__new__(AGSSWEEnvironment)
        environment.tcr = TCRConfig("registry.example", "cache", "user", "secret")
        environment.sandbox = SimpleNamespace(files=SimpleNamespace(write=Mock()))
        return environment

    @patch("swe_rl.sandbox.ags.time.sleep")
    def test_login_retries_transient_failure(self, sleep: Mock):
        environment = self._environment()
        environment._outer_run = Mock(
            side_effect=[
                CommandResult("timeout", 1, 60),
                CommandResult("Login Succeeded", 0, 1),
                CommandResult("", 0, 0),
            ]
        )

        self.assertTrue(environment._docker_login())
        self.assertEqual(environment._outer_run.call_count, 3)
        sleep.assert_called_once_with(2)

    def test_tcr_outage_falls_back_to_upstream(self):
        environment = self._environment()
        environment._docker_login = Mock(return_value=False)
        environment._outer_run = Mock(return_value=CommandResult("pulled", 0, 1))

        self.assertEqual(environment._pull_image("swebench/task:latest"), "swebench/task:latest")
        command = environment._outer_run.call_args.args[0]
        self.assertEqual(command, "sudo docker pull swebench/task:latest")


class CommandTimeoutTests(unittest.TestCase):
    def test_outer_wall_clock_timeout_terminates_sandbox(self):
        release = threading.Event()
        sandbox = SimpleNamespace(
            sandbox_id="sandbox-test",
            commands=SimpleNamespace(run=Mock(side_effect=lambda *args, **kwargs: release.wait(10))),
            kill=Mock(side_effect=release.set),
        )
        environment = object.__new__(AGSSWEEnvironment)
        environment.task = {"instance_id": "example__task-1"}
        environment.sandbox = sandbox
        environment.invalid_reason = ""
        environment.heartbeat_seconds = 0.01
        environment.wall_clock_grace_seconds = 0.0

        started = time.monotonic()
        result = environment._outer_run("sleep forever", timeout=0.02, category="test")

        self.assertEqual(result.return_code, 124)
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertIsNone(environment.sandbox)
        self.assertTrue(release.wait(1.0))
        self.assertIn("wall-clock", environment.invalid_reason)

    def test_close_does_not_wait_forever_for_sdk_kill(self):
        release = threading.Event()
        sandbox = SimpleNamespace(
            sandbox_id="sandbox-close-test",
            kill=Mock(side_effect=lambda: release.wait(10)),
        )
        environment = object.__new__(AGSSWEEnvironment)
        environment.task = {"instance_id": "example__task-2"}
        environment.sandbox = sandbox
        environment.sandbox_kill_timeout_seconds = 0.02
        environment._outer_run = Mock(return_value=CommandResult("", 0, 0.0))

        started = time.monotonic()
        environment.close()

        self.assertLess(time.monotonic() - started, 1.0)
        self.assertIsNone(environment.sandbox)
        release.set()


class TestTimeoutPropagationTests(unittest.TestCase):
    def test_run_tests_uses_requested_timeout(self):
        environment = object.__new__(AGSSWEEnvironment)
        environment.task = {
            "test_patch": "",
            "test_command": "pytest -q",
            "fail_to_pass": [],
            "pass_to_pass": [],
        }
        environment.sandbox = SimpleNamespace()
        environment.execute = Mock(return_value=CommandResult("", 1, 2.0))

        with patch("swe_rl.sandbox.ags.score_test_output", return_value=Mock()) as score:
            environment.run_tests(timeout=321)

        environment.execute.assert_called_once_with(
            "pytest -q", timeout=321, truncate=False, category="official_tests"
        )
        score.assert_called_once()


if __name__ == "__main__":
    unittest.main()
