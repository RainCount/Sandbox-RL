from __future__ import annotations

import unittest

from swe_rl.agent.protocol import TOOLS, build_initial_messages, extract_action


class ProtocolTests(unittest.TestCase):
    def test_extracts_one_command(self):
        action, error = extract_action("Reasoning\n```bash\npytest -q\n```")
        self.assertEqual(action, "pytest -q")
        self.assertIsNone(error)

    def test_extracts_native_qwen_bash_tool_call(self):
        response = (
            '<tool_call>\n{"name":"bash","arguments":'
            '{"command":"sed -n \'1,20p\' x.py"}}\n</tool_call>'
        )
        action, error = extract_action(response)
        self.assertEqual(action, "sed -n '1,20p' x.py")
        self.assertIsNone(error)

    def test_accepts_sera_argument_variants(self):
        direct = '<tool_call>{"name":"bash","arguments":"ls -la"}</tool_call>'
        python_dict = (
            '<tool_call>{"name":"bash","arguments":'
            '"{\'command\': \'grep -n x y.py\'}"}</tool_call>'
        )
        self.assertEqual(extract_action(direct), ("ls -la", None))
        self.assertEqual(extract_action(python_dict), ("grep -n x y.py", None))

    def test_translates_native_str_replace_editor(self):
        response = (
            '<tool_call>{"name":"str_replace_editor","arguments":'
            '{"command":"str_replace","path":"x.py","old_str":"x = 1",'
            '"new_str":"x = 2"}}</tool_call>'
        )
        action, error = extract_action(response)
        self.assertIsNone(error)
        self.assertEqual(
            action,
            "STR_REPLACE x.py\n<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE",
        )

    def test_translates_native_editor_view(self):
        response = (
            '<tool_call>{"name":"str_replace_editor","arguments":'
            '{"command":"view","path":"/testbed/x.py","view_range":[10,20]}}</tool_call>'
        )
        action, error = extract_action(response)
        self.assertIsNone(error)
        self.assertEqual(
            action,
            'STR_VIEW {"path": "/testbed/x.py", "view_range": [10, 20]}',
        )

    def test_accepts_sera_view_range_command_alias(self):
        response = (
            '<tool_call>{"name":"str_replace_editor","arguments":'
            '{"command":"view_range","path":"/testbed/x.py",'
            '"start_line":10,"end_line":20}}</tool_call>'
        )
        self.assertEqual(
            extract_action(response),
            ('STR_VIEW {"path": "/testbed/x.py", "view_range": [10, 20]}', None),
        )

    def test_translates_native_editor_create_and_delete(self):
        create = (
            '<tool_call>{"name":"str_replace_editor","arguments":'
            '{"command":"create","path":"/testbed/x.py","file_text":"x = 1\\n"}}'
            "</tool_call>"
        )
        self.assertEqual(
            extract_action(create),
            ('STR_CREATE {"path": "/testbed/x.py", "file_text": "x = 1\\n"}', None),
        )
        delete = (
            '<tool_call>{"name":"str_replace_editor","arguments":'
            '{"command":"str_replace","path":"/testbed/x.py","old_str":"x = 1"}}'
            "</tool_call>"
        )
        self.assertEqual(
            extract_action(delete),
            (
                "/testbed/x.py\n<<<<<<< SEARCH\nx = 1\n=======\n\n>>>>>>> REPLACE".replace(
                    "/testbed", "STR_REPLACE /testbed", 1
                ),
                None,
            ),
        )

    def test_editor_schema_matches_sera_commands(self):
        editor = next(tool for tool in TOOLS if tool["function"]["name"] == "str_replace_editor")
        commands = editor["function"]["parameters"]["properties"]["command"]["enum"]
        self.assertEqual(commands, ["view", "create", "str_replace"])

    def test_accepts_native_submit_tool_call(self):
        response = '<tool_call>{"name":"submit","arguments":{}}</tool_call>'
        self.assertEqual(extract_action(response), ("submit", None))

    def test_reports_unclosed_native_tool_call_as_truncated(self):
        action, error = extract_action('<tool_call>{"name":"str_replace_editor"')
        self.assertIsNone(action)
        self.assertIn("truncated tool call", error)

    def test_uses_first_fence_from_legacy_replayed_trajectory(self):
        action, error = extract_action("```bash\nls\n```\n```bash\nfake output\n```")
        self.assertEqual(action, "ls")
        self.assertIsNone(error)

    def test_rejects_placeholder(self):
        self.assertIsNotNone(extract_action("```bash\nyour_command_here\n```")[1])

    def test_prompt_contains_issue_and_tests(self):
        messages = build_initial_messages(
            {
                "repo": "org/repo",
                "instance_id": "org__repo-1",
                "problem_statement": "Fix the bug",
                "fail_to_pass": ["tests/test_bug.py::test_one"],
            }
        )
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Fix the bug", messages[1]["content"])
        self.assertIn("test_one", messages[1]["content"])
        self.assertIn("CRITICAL RULES", messages[0]["content"])
        self.assertIn("str_replace_editor", messages[0]["content"])
        self.assertIn("smallest unique expression", messages[0]["content"])
        self.assertIn("do NOT retry the same", messages[0]["content"])
        self.assertIn("verify the fix", messages[0]["content"])
        self.assertIn("prefer `grep -Rsn", messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
