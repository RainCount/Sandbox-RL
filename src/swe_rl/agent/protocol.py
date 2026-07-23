"""SERA/Qwen-compatible tool protocol for coding-agent rollouts."""

from __future__ import annotations

import ast
import json
import re
from typing import Any

SYSTEM_PROMPT = """You are an expert software engineer resolving a real GitHub issue in a Python
repository. You interact with the repository by making exactly ONE tool call per response using
the format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>.

CRITICAL RULES — violating any of these will cause the task to fail:
1. You MUST make at least one str_replace_editor `str_replace` call that changes production source
   code. Reading code without editing will NOT solve the issue — tests will run automatically and
   you will get zero credit if no source files were changed.
2. Before `str_replace`, read the exact target line(s) with `view` + `view_range` and copy the
   output VERBATIM into `old_str`. Never guess or reconstruct `old_str` from memory.
3. Keep `old_str` small: replace 1-8 lines (the smallest unique expression). Never send an entire
   function, class, or docstring — large replacements almost always fail to match.
4. After your edit, run at least one focused test or reproduction script to verify the fix.
5. When you are confident the fix is correct, call the `submit` tool. Do NOT keep searching
   after verification passes. A patch is useless unless submitted.

WORKFLOW:
  Phase 1 (Explore): Use `grep -n`/`find` to locate the relevant production symbol, then
    `str_replace_editor` `view` with `view_range` to read only the necessary 20-60 lines.
  Phase 2 (Reproduce): Run a minimal reproduction script or the failing test to confirm the bug.
  Phase 3 (Edit): Read the exact target, then make one `str_replace_editor` `str_replace` call.
  Phase 4 (Verify): Re-run the test/reproducer. If it fails, read the exact target again and
    make a SMALLER, more precise edit — do NOT retry the same guessed replacement.
  Phase 5 (Submit): Call `submit` once verification passes. Then STOP.

Always use absolute `/testbed/...` paths. Do NOT edit test files. Put scratch scripts under
`/tmp`, not inside `/testbed`. Never install packages with pip/apt/conda — the environment
is fixed. Use `rg` only if confirmed available; prefer `grep -Rsn --include='*.py'`.
"""


_ACTION_BLOCK = re.compile(r"```(?:bash|sh|mswea_bash_command)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL | re.IGNORECASE)
_PLACEHOLDERS = {"your_command_here", "one command", "exactly_one_command_here", "command"}

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Execute one non-interactive bash command and return stdout/stderr. "
            "Commands run in a persistent shell: environment variables and working directory "
            "carry over between calls. Use && to chain related commands. Avoid commands that "
            "produce huge output (>100 lines); pipe through '| tail -n 50' or '| head -n 50'. "
            "For background servers use 'command &' and note the PID. "
            "You do NOT have internet access — only the local testbed environment is available."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to execute."}
            },
            "required": ["command"],
        },
    },
}

STR_REPLACE_EDITOR_TOOL = {
    "type": "function",
    "function": {
        "name": "str_replace_editor",
        "description": (
            "File editor for viewing, creating, and editing source files. "
            "State persists across calls — the file retains your previous edits.\n"
            "Commands:\n"
            "  view — show file content. For large files use view_range=[start, end] "
            "to limit output. Long output is truncated with a warning.\n"
            "  create — write a new file. The destination path must NOT already exist.\n"
            "  str_replace — replace old_str with new_str in the file. "
            "CRITICAL: old_str must EXACTLY match one or more consecutive lines "
            "from the current file content. Copy it character-for-character from a "
            "previous view output. Whitespace and indentation matter. "
            "If the match is not unique, the edit is rejected — include more context "
            "lines to disambiguate. Keep old_str to 1-8 lines for highest success rate."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "enum": ["view", "create", "str_replace"]},
                "path": {
                    "type": "string",
                    "description": "Absolute file or directory path under /testbed.",
                },
                "file_text": {
                    "type": "string",
                    "description": "Required for create: complete content of the new file.",
                },
                "old_str": {
                    "type": "string",
                    "description": "Required for str_replace: exact text to replace (1-8 lines).",
                },
                "new_str": {
                    "type": "string",
                    "description": "Required for str_replace: replacement text. Empty string to delete.",
                },
                "view_range": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional for view: [start_line, end_line] inclusive. end=-1 for EOF.",
                },
            },
            "required": ["command", "path"],
        },
    },
}

SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit",
        "description": (
            "Submit your source-code changes for official evaluation. "
            "Call this ONLY after you have made at least one str_replace_editor str_replace "
            "call AND verified the fix works. After calling submit, the task ends — "
            "do NOT make any further tool calls."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

TOOLS = [BASH_TOOL, STR_REPLACE_EDITOR_TOOL, SUBMIT_TOOL]


def _decode_arguments(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    for decoder in (json.loads, ast.literal_eval):
        try:
            return decoder(value)
        except (json.JSONDecodeError, SyntaxError, ValueError):
            pass
    # Some SERA generations put the command directly in ``arguments``.
    return value


def extract_action(response: str) -> tuple[str | None, str | None]:
    """Extract one action from native Qwen tool calls or legacy fenced output."""
    response = response or ""
    calls = _TOOL_CALL.findall(response)
    if calls:
        try:
            payload = json.loads(calls[0])
            function = payload.get("function", {})
            name = payload.get("name") or function.get("name")
            arguments = payload.get("arguments", function.get("arguments", {}))
            arguments = _decode_arguments(arguments)
            if name == "bash":
                action = arguments.get("command") if isinstance(arguments, dict) else arguments
            elif name == "submit":
                action = "submit"
            elif name == "str_replace_editor" and isinstance(arguments, dict):
                command = arguments.get("command")
                path = arguments.get("path")
                if not isinstance(path, str) or not path.strip():
                    return None, "FORMAT_ERROR: str_replace_editor path is required"
                # A minority of SERA trajectories use ``view_range`` as the
                # command name even though the released tool schema models it
                # as ``command=view`` plus a ``view_range`` argument. Accept
                # that learned alias so one malformed choice cannot consume
                # every remaining turn.
                if command in {"view", "view_range"}:
                    view_range = arguments.get("view_range")
                    if view_range is None and command == "view_range":
                        start = arguments.get("start_line")
                        end = arguments.get("end_line")
                        if isinstance(start, int) and isinstance(end, int):
                            view_range = [start, end]
                    if view_range is not None and (
                        not isinstance(view_range, list)
                        or len(view_range) != 2
                        or not all(isinstance(value, int) for value in view_range)
                    ):
                        return None, "FORMAT_ERROR: view_range must be [start, end]"
                    action = "STR_VIEW " + json.dumps(
                        {"path": path, "view_range": view_range}, ensure_ascii=False
                    )
                elif command == "create":
                    file_text = arguments.get("file_text")
                    if not isinstance(file_text, str):
                        return None, "FORMAT_ERROR: create requires file_text"
                    action = "STR_CREATE " + json.dumps(
                        {"path": path, "file_text": file_text}, ensure_ascii=False
                    )
                elif command == "str_replace":
                    old = arguments.get("old_str")
                    new = arguments.get("new_str", "")
                    if not all(isinstance(value, str) for value in (old, new)):
                        return None, "FORMAT_ERROR: str_replace requires old_str and string new_str"
                    action = (
                        f"STR_REPLACE {path}\n<<<<<<< SEARCH\n{old}\n=======\n{new}\n"
                        ">>>>>>> REPLACE"
                    )
                else:
                    return None, f"FORMAT_ERROR: unsupported str_replace_editor command {command!r}"
            else:
                return None, f"FORMAT_ERROR: unsupported tool {name!r}"
            if not isinstance(action, str) or not action.strip():
                return None, "FORMAT_ERROR: expected a non-empty tool action"
            action = action.strip()
            if action.lower() in _PLACEHOLDERS:
                return None, "FORMAT_ERROR: placeholder commands are not executable"
            return action, None
        except (json.JSONDecodeError, AttributeError, TypeError):
            return None, "FORMAT_ERROR: malformed tool call JSON"

    if "<tool_call" in response.lower() and "</tool_call>" not in response.lower():
        return (
            None,
            "FORMAT_ERROR: truncated tool call; do not retry a long edit. "
            "Read exact lines and replace only 1-8 lines",
        )

    blocks = _ACTION_BLOCK.findall(response)
    if not blocks:
        return None, "FORMAT_ERROR: expected one bash tool call or fenced command"
    action = blocks[0].strip()
    if not action:
        return None, "FORMAT_ERROR: empty command"
    if action.lower() in _PLACEHOLDERS:
        return None, "FORMAT_ERROR: placeholder commands are not executable"
    return action, None


def build_initial_messages(task: dict[str, Any]) -> list[dict[str, str]]:
    fail_tests = (
        "\n".join(f"- {test}" for test in task.get("fail_to_pass", []))
        or "- (read the issue description to identify failing behavior)"
    )
    user = (
        f"/testbed\n"
        f"I've uploaded a python code repository in /testbed.\n\n"
        f"<issue>\n{task['problem_statement']}\n</issue>\n\n"
        f"Failing tests:\n{fail_tests}\n\n"
        "Your task is to implement the necessary changes to the repository so that "
        "the issue is resolved. Make minimal changes to non-test files.\n\n"
        "IMPORTANT: You MUST make at least one str_replace_editor `str_replace` edit. "
        "Reading code without editing will NOT solve the issue — you will get zero credit."
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
