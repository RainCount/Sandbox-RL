"""Run one official SWE-bench instance image inside an AGS CPU sandbox.

The AGS template owns Docker/fuse-overlayfs. The task image owns the exact
repository and Python dependencies. The trainer never executes untrusted code
on the TKE host.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import PurePosixPath
import queue
import re
import shlex
import threading
import time
from dataclasses import dataclass
from typing import Any

from swe_rl.config import AGSConfig, TCRConfig
from swe_rl.sandbox.bootstrap import (
    STOP_DOCKERD_COMMAND,
    dockerd_command,
    ensure_fuse_overlayfs,
)
from swe_rl.schema import RewardResult

logger = logging.getLogger(__name__)
# Ray workers commonly leave the root logger at WARNING. Keep this narrow
# operational stream visible without changing verbosity for any other module.
logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class CommandResult:
    output: str
    return_code: int
    duration_seconds: float


def _truncate_observation(value: str, maximum: int = 4_000, maximum_lines: int = 120) -> str:
    lines = value.splitlines(keepends=True)
    if len(lines) > maximum_lines:
        half = maximum_lines // 2
        omitted = len(lines) - maximum_lines
        value = "".join(
            [*lines[:half], f"\n... {omitted} lines omitted ...\n", *lines[-half:]]
        )
    if len(value) <= maximum:
        return value
    head = maximum // 2
    tail = maximum - head
    return f"{value[:head]}\n... {len(value) - maximum} characters omitted ...\n{value[-tail:]}"


def _modified_files(patch: str) -> list[str]:
    matches = re.findall(r"^diff --git a/(.+?) b/(.+)$", patch, flags=re.MULTILINE)
    return sorted({destination for _source, destination in matches})


def _validate_editor_path(path: str) -> str | None:
    """Return a user-facing error when an editor path escapes the task repository."""
    candidate = PurePosixPath(path)
    if ".." in candidate.parts:
        return "Editor paths must not contain '..'"
    if candidate.is_absolute():
        if path != "/testbed" and not path.startswith("/testbed/"):
            return "Editor paths must stay under /testbed"
    return None


def _is_top_level_test_path(path: str) -> bool:
    relative = path.removeprefix("/testbed/").removeprefix("./")
    parts = PurePosixPath(relative).parts
    if not parts:
        return False
    first = parts[0].lower()
    if first in {"test", "tests", "testing"}:
        return True
    # SERA sometimes creates a root-level helper and then mistakes that
    # successful CREATE for a production edit.  Such files can also be picked
    # up by broad pytest collection. Scratch reproducers belong under /tmp.
    if len(parts) == 1:
        filename = PurePosixPath(first).name
        return filename.startswith(("test_", "test-", "repro", "check_"))
    return False


class AGSSWEEnvironment:
    container_name = "swe-testbed"
    repo_dir = "/testbed"
    heartbeat_seconds = 30.0
    wall_clock_grace_seconds = 15.0
    sandbox_kill_timeout_seconds = 15.0

    def __init__(self, task: dict[str, Any], ags: AGSConfig, tcr: TCRConfig):
        self.task = task
        self.ags = ags
        self.tcr = tcr
        self.sandbox = None
        self.invalid_reason = ""
        self.run_image = ""
        self.run_image_digest = ""

    @property
    def sandbox_id(self) -> str:
        if self.sandbox is None:
            return ""
        return str(
            getattr(self.sandbox, "sandbox_id", "")
            or getattr(self.sandbox, "id", "")
            or "unknown"
        )

    def _log_event(self, event: str, **fields: Any) -> None:
        payload = {
            "event": event,
            "instance_id": str(self.task.get("instance_id", "unknown")),
            "sandbox_id": self.sandbox_id,
            **fields,
        }
        logger.info("ags_event=%s", json.dumps(payload, ensure_ascii=True, sort_keys=True))

    def _kill_sandbox(self, sandbox: Any, *, wait_seconds: float) -> None:
        finished = threading.Event()
        errors: list[str] = []
        sandbox_id = str(
            getattr(sandbox, "sandbox_id", "") or getattr(sandbox, "id", "") or "unknown"
        )

        def kill() -> None:
            try:
                sandbox.kill()
            except BaseException as exc:
                errors.append(str(exc))
            finally:
                finished.set()

        threading.Thread(target=kill, name="ags-sandbox-kill", daemon=True).start()
        if wait_seconds <= 0:
            return
        if not finished.wait(wait_seconds):
            self._log_event(
                "sandbox_kill_timed_out",
                sandbox_id=sandbox_id,
                timeout_seconds=wait_seconds,
            )
        elif errors:
            self._log_event("sandbox_kill_failed", sandbox_id=sandbox_id, error=errors[0])

    def __enter__(self) -> AGSSWEEnvironment:
        os.environ["E2B_DOMAIN"] = self.ags.domain
        os.environ["E2B_API_KEY"] = self.ags.api_key
        from e2b_code_interpreter import Sandbox

        try:
            self.sandbox = Sandbox(
                template=self.ags.template,
                timeout=self.ags.timeout_seconds,
                proxy=self.ags.proxy_url or None,
            )
            self._log_event("sandbox_started", template=self.ags.template)
            self._start_docker()
            self.run_image = self._pull_image(str(self.task["docker_image"]))
            self.run_image_digest = self._image_digest(self.run_image)
            self._start_container()
            self._install_test_patch()
            return self
        except BaseException:
            self.close()
            raise

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _outer_run(
        self,
        command: str,
        timeout: float = 120,
        user: str = "user",
        *,
        category: str = "sandbox_command",
    ) -> CommandResult:
        if self.sandbox is None:
            raise RuntimeError("sandbox is not started")
        sandbox = self.sandbox
        started = time.monotonic()
        command_id = f"{int(started * 1000):x}-{threading.get_ident():x}"
        self._log_event(
            "command_started",
            command_id=command_id,
            category=category,
            timeout_seconds=timeout,
        )
        outcome: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

        def invoke() -> None:
            try:
                outcome.put(("result", sandbox.commands.run(command, timeout=timeout, user=user)))
            except BaseException as exc:  # SDK exceptions must be returned to the rollout thread.
                outcome.put(("exception", exc))

        worker = threading.Thread(target=invoke, name=f"ags-{category}", daemon=True)
        worker.start()
        wall_timeout = max(float(timeout), 0.0) + max(float(self.wall_clock_grace_seconds), 0.0)
        deadline = started + wall_timeout
        heartbeat = max(float(self.heartbeat_seconds), 0.1)
        item: tuple[str, Any] | None = None
        while item is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                item = outcome.get(timeout=min(heartbeat, remaining))
            except queue.Empty:
                self._log_event(
                    "command_waiting",
                    command_id=command_id,
                    category=category,
                    elapsed_seconds=round(time.monotonic() - started, 3),
                )

        if item is None:
            duration = time.monotonic() - started
            self._log_event(
                "command_timed_out",
                command_id=command_id,
                category=category,
                duration_seconds=round(duration, 3),
                return_code=124,
            )
            # A broken SDK call must not pin the rollout forever. Killing the
            # remote sandbox also gives the daemon worker a chance to unwind.
            self._kill_sandbox(sandbox, wait_seconds=0)
            self.invalid_reason = f"{category} exceeded its wall-clock limit"
            if self.sandbox is sandbox:
                self.sandbox = None
            return CommandResult(
                f"TIMEOUT: {category} exceeded {wall_timeout:.1f}s wall-clock limit",
                124,
                duration,
            )

        kind, value = item
        duration = time.monotonic() - started
        try:
            if kind == "exception":
                raise value
            result = value
            stdout = getattr(result, "stdout", "") or ""
            stderr = getattr(result, "stderr", "") or ""
            code = getattr(result, "exit_code", getattr(result, "exitCode", 0)) or 0
            command_result = CommandResult(stdout + stderr, int(code), duration)
        except Exception as exc:
            stdout = getattr(exc, "stdout", "") or ""
            stderr = getattr(exc, "stderr", "") or ""
            message = stdout + stderr or str(exc)
            match = re.search(r"exit(?:ed)?(?: with)? code[:\s]+(\d+)", str(exc), re.IGNORECASE)
            command_result = CommandResult(message, int(match.group(1)) if match else 1, duration)
        self._log_event(
            "command_finished",
            command_id=command_id,
            category=category,
            duration_seconds=round(command_result.duration_seconds, 3),
            return_code=command_result.return_code,
        )
        return command_result

    def _start_docker(self) -> None:
        available, _installed = ensure_fuse_overlayfs(self.sandbox)
        if not available:
            raise RuntimeError("could not install the pinned fuse-overlayfs controller asset")
        stopped = self._outer_run(STOP_DOCKERD_COMMAND, timeout=20)
        if stopped.return_code != 0:
            raise RuntimeError(f"could not stop template dockerd: {stopped.output[-1000:]}")
        self._outer_run("sudo rm -rf /var/lib/docker/*", timeout=30)
        self._outer_run(dockerd_command(self.ags.dockerhub_mirror), timeout=10)
        for _ in range(12):
            result = self._outer_run(
                "sudo docker info >/dev/null 2>&1 && echo READY || echo WAIT", timeout=10
            )
            if "READY" in result.output:
                return
            time.sleep(2)
        logs = self._outer_run("sudo tail -50 /tmp/dockerd.log", timeout=10).output
        raise RuntimeError(f"dockerd did not become ready: {logs[-3000:]}")

    def _docker_login(self, attempts: int = 3) -> bool:
        if not (self.tcr.registry and self.tcr.username and self.tcr.password):
            return False
        assert self.sandbox is not None
        password_path = "/tmp/tcr-password"
        self.sandbox.files.write(password_path, self.tcr.password.encode())
        try:
            for attempt in range(1, attempts + 1):
                result = self._outer_run(
                    f"cat {password_path} | sudo docker login {shlex.quote(self.tcr.registry)} "
                    f"--username {shlex.quote(self.tcr.username)} --password-stdin",
                    timeout=60,
                )
                if result.return_code == 0:
                    return True
                if attempt < attempts:
                    delay = 2 ** attempt
                    logger.warning(
                        "TCR login attempt %d/%d failed; retrying in %d seconds",
                        attempt,
                        attempts,
                        delay,
                    )
                    time.sleep(delay)
            logger.warning(
                "TCR login failed after %d attempts; continuing with upstream image sources",
                attempts,
            )
            return False
        finally:
            self._outer_run(f"rm -f {password_path}", timeout=10)

    def _pull_image(self, upstream: str) -> str:
        tcr_available = self._docker_login()
        if tcr_available:
            cached = f"{self.tcr.registry}/{self.tcr.namespace}/{upstream}"
            result = self._outer_run(f"sudo docker pull {shlex.quote(cached)}", timeout=900)
            if result.return_code == 0:
                return cached
        # Docker Hub mirrors implement the Registry mirror protocol. They are not
        # ordinary registries, so keep the upstream image name and let dockerd route it.
        source_image = upstream
        result = self._outer_run(f"sudo docker pull {shlex.quote(source_image)}", timeout=900)
        errors = [f"dockerhub: {result.output[-1200:]}"] if result.return_code != 0 else []
        if result.return_code != 0:
            for prefix in self.ags.dockerhub_fallback_prefixes:
                source_image = f"{prefix}/{upstream.removeprefix('docker.io/')}"
                result = self._outer_run(f"sudo docker pull {shlex.quote(source_image)}", timeout=900)
                if result.return_code == 0:
                    break
                errors.append(f"{prefix}: {result.output[-1200:]}")
        if result.return_code != 0:
            raise RuntimeError("SWE-bench image pull failed:\n" + "\n".join(errors)[-4000:])
        if tcr_available:
            cached = f"{self.tcr.registry}/{self.tcr.namespace}/{upstream}"
            tagged = self._outer_run(
                f"sudo docker tag {shlex.quote(source_image)} {shlex.quote(cached)}", timeout=60
            )
            if tagged.return_code == 0:
                # Cache population is an optimization; rollout can continue if push is unavailable.
                self._outer_run(f"sudo docker push {shlex.quote(cached)}", timeout=900)
        return source_image

    def _image_digest(self, image: str) -> str:
        result = self._outer_run(
            "sudo docker image inspect --format '{{json .RepoDigests}}' " + shlex.quote(image),
            timeout=30,
        )
        return result.output.strip() if result.return_code == 0 else ""

    def _start_container(self) -> None:
        self._outer_run(f"sudo docker rm -f {self.container_name} 2>/dev/null || true", timeout=30)
        result = self._outer_run(
            f"sudo docker run -d --name {self.container_name} "
            "-e BASH_ENV=/root/.bashrc -e PAGER=cat -e MANPAGER=cat -e GIT_PAGER=cat "
            f"--entrypoint tail {shlex.quote(self.run_image)} -f /dev/null",
            timeout=120,
        )
        if result.return_code != 0:
            raise RuntimeError(f"SWE-bench container failed to start: {result.output[-2000:]}")

    def _install_test_patch(self) -> None:
        patch = str(self.task.get("test_patch", ""))
        if not patch:
            return
        assert self.sandbox is not None
        self.sandbox.files.write("/tmp/test.patch", patch.encode())
        copied = self._outer_run(
            f"sudo docker cp /tmp/test.patch {self.container_name}:/tmp/test.patch", timeout=30
        )
        if copied.return_code != 0:
            raise RuntimeError(f"could not copy test patch: {copied.output}")
        command = (
            f"cd {self.repo_dir} && git config user.email swe-rl@localhost && git config user.name swe-rl "
            "&& git apply --check /tmp/test.patch && git apply /tmp/test.patch "
            "&& git add -A && git commit --no-gpg-sign -m 'swe-rl test patch'"
        )
        result = self.execute(command, timeout=60, truncate=False)
        if result.return_code != 0:
            raise RuntimeError(f"test patch did not apply cleanly: {result.output[-3000:]}")

    def _replace(self, command: str, timeout: int) -> CommandResult | None:
        if not command.lstrip().startswith("STR_REPLACE "):
            return None
        first, _, body = command.strip().partition("\n")
        path = first.split(maxsplit=1)[1].strip()
        path_error = _validate_editor_path(path)
        if path_error:
            return CommandResult(path_error, 2, 0.0)
        if _is_top_level_test_path(path):
            return CommandResult("TEST_EDIT_REFUSED: edit production source, never tests", 2, 0.0)
        if not all(marker in body for marker in ("<<<<<<< SEARCH", "=======", ">>>>>>> REPLACE")):
            return CommandResult("STR_REPLACE requires SEARCH/REPLACE markers", 2, 0.0)
        search = body.split("<<<<<<< SEARCH", 1)[1].split("=======", 1)[0].strip("\n")
        replacement = body.split("=======", 1)[1].split(">>>>>>> REPLACE", 1)[0].strip("\n")
        payload = base64.b64encode(search.encode()).decode()
        new_payload = base64.b64encode(replacement.encode()).decode()
        script = (
            "import ast,base64,pathlib,sys;"
            f"p=pathlib.Path({path!r});s=base64.b64decode({payload!r}).decode();"
            f"r=base64.b64decode({new_payload!r}).decode();"
            "x=p.read_text(encoding='utf-8');n=x.count(s);"
            "n==1 or sys.exit('SEARCH must match exactly once');y=x.replace(s,r);"
            "ast.parse(y,filename=str(p)) if p.suffix=='.py' else None;"
            "p.write_text(y,encoding='utf-8');print('STR_REPLACE_OK')"
        )
        encoded = base64.b64encode(script.encode()).decode()
        return self.execute(
            f"echo {encoded} | base64 -d | python", timeout=timeout, truncate=False, intercept=False
        )

    def _view(self, command: str, timeout: int) -> CommandResult | None:
        if not command.lstrip().startswith("STR_VIEW "):
            return None
        try:
            arguments = json.loads(command.strip()[len("STR_VIEW ") :])
            path = arguments["path"]
            view_range = arguments.get("view_range")
        except (json.JSONDecodeError, KeyError, TypeError):
            return CommandResult("STR_VIEW requires valid JSON arguments", 2, 0.0)
        if not isinstance(path, str) or not path:
            return CommandResult("STR_VIEW path is required", 2, 0.0)
        path_error = _validate_editor_path(path)
        if path_error:
            return CommandResult(path_error, 2, 0.0)
        quoted = shlex.quote(path)
        if view_range is not None:
            if (
                not isinstance(view_range, list)
                or len(view_range) != 2
                or not all(isinstance(value, int) for value in view_range)
            ):
                return CommandResult("STR_VIEW range must be [start, end]", 2, 0.0)
            start, end = view_range
            if start < 1 or (end != -1 and end < start):
                return CommandResult("STR_VIEW range is invalid", 2, 0.0)
            range_expression = f"{start},$p" if end == -1 else f"{start},{end}p"
            file_command = f"nl -ba -- {quoted} | sed -n {shlex.quote(range_expression)}"
        else:
            file_command = f"nl -ba -- {quoted}"
        shell = (
            f"if [ -d {quoted} ]; then "
            f"find {quoted} -maxdepth 2 -not -path '*/.*' | sort | head -100; "
            f"elif [ -f {quoted} ]; then {file_command}; "
            "else echo 'STR_VIEW path does not exist' >&2; exit 2; fi"
        )
        return self.execute(shell, timeout=timeout, intercept=False)

    def _create(self, command: str, timeout: int) -> CommandResult | None:
        if not command.lstrip().startswith("STR_CREATE "):
            return None
        try:
            arguments = json.loads(command.strip()[len("STR_CREATE ") :])
            path = arguments["path"]
            file_text = arguments["file_text"]
        except (json.JSONDecodeError, KeyError, TypeError):
            return CommandResult("STR_CREATE requires valid JSON arguments", 2, 0.0)
        if not isinstance(path, str) or not isinstance(file_text, str):
            return CommandResult("STR_CREATE requires string path and file_text", 2, 0.0)
        path_error = _validate_editor_path(path)
        if path_error:
            return CommandResult(path_error, 2, 0.0)
        if _is_top_level_test_path(path):
            return CommandResult("TEST_EDIT_REFUSED: create production source, never tests", 2, 0.0)
        payload = base64.b64encode(file_text.encode()).decode()
        script = (
            "import ast,base64,pathlib,sys;"
            f"p=pathlib.Path({path!r});p.exists() and sys.exit('CREATE path already exists');"
            f"x=base64.b64decode({payload!r}).decode();"
            "ast.parse(x,filename=str(p)) if p.suffix=='.py' else None;"
            "p.parent.mkdir(parents=True,exist_ok=True);"
            "p.write_text(x,encoding='utf-8');print('STR_CREATE_OK')"
        )
        encoded = base64.b64encode(script.encode()).decode()
        return self.execute(
            f"echo {encoded} | base64 -d | python && git add -N -- {shlex.quote(path)}",
            timeout=timeout,
            truncate=False,
            intercept=False,
        )

    def execute(
        self,
        command: str,
        *,
        timeout: int = 120,
        truncate: bool = True,
        intercept: bool = True,
        category: str = "task_command",
    ) -> CommandResult:
        if intercept:
            for editor in (self._view, self._create, self._replace):
                result = editor(command, timeout)
                if result is not None:
                    return result
        encoded = base64.b64encode(command.encode()).decode()
        outer = (
            f"echo {encoded} | base64 -d | sudo docker exec -i {self.container_name} "
            f"bash -c 'cd {self.repo_dir} && bash'"
        )
        result = self._outer_run(outer, timeout=timeout, category=category)
        output = _truncate_observation(result.output) if truncate else result.output
        return CommandResult(output, result.return_code, result.duration_seconds)

    def patch(self) -> str:
        if self.sandbox is None:
            return ""
        return self.execute(
            "git -c core.fileMode=false diff --binary HEAD",
            timeout=30,
            truncate=False,
            category="repository_diff",
        ).output

    def has_source_changes(self) -> bool:
        return bool(self.patch().strip())

    def run_tests(self, *, timeout: int = 900) -> tuple[RewardResult, str]:
        if self.sandbox is None:
            output = f"INFRA_ERROR: {self.invalid_reason or 'sandbox is unavailable'}"
            return (
                RewardResult(
                    reward=0.0,
                    resolved=False,
                    fail_to_pass_success=0,
                    fail_to_pass_total=len(self.task.get("fail_to_pass", [])),
                    pass_to_pass_success=0,
                    pass_to_pass_total=len(self.task.get("pass_to_pass", [])),
                    test_exit_code=124,
                    parser_ok=False,
                    raw_tail=output,
                ),
                output,
            )
        test_patch = str(self.task.get("test_patch", ""))
        test_files = _modified_files(test_patch)
        if test_files:
            quoted = " ".join(shlex.quote(path) for path in test_files)
            self.execute(f"git checkout HEAD -- {quoted}", timeout=30, truncate=False)
        command = str(self.task["test_command"])
        result = self.execute(
            command, timeout=timeout, truncate=False, category="official_tests"
        )
        reward = score_test_output(self.task, result.output, result.return_code)
        return reward, result.output

    def close(self) -> None:
        if self.sandbox is None:
            return
        sandbox = self.sandbox
        sandbox_id = self.sandbox_id
        self._log_event("sandbox_closing", sandbox_id=sandbox_id)
        try:
            self._outer_run(f"sudo docker rm -f {self.container_name}", timeout=30)
        finally:
            self.sandbox = None
            self._kill_sandbox(sandbox, wait_seconds=self.sandbox_kill_timeout_seconds)
            self._log_event("sandbox_closed", sandbox_id=sandbox_id)


def score_test_output(task: dict[str, Any], output: str, exit_code: int) -> RewardResult:
    """Use SWE-bench's official repository-specific parser and grading rules."""
    try:
        from swebench.harness.constants import (
            FAIL_ONLY_REPOS,
            FAIL_TO_PASS,
            PASS_TO_PASS,
            EvalType,
            ResolvedStatus,
        )
        from swebench.harness.grading import (
            compute_pass_to_pass,
            get_eval_tests_report,
            get_resolution_status,
        )
        from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
        from swebench.harness.test_spec.test_spec import make_test_spec

        instance = {
            "instance_id": task["instance_id"],
            "repo": task["repo"],
            "version": task["version"],
            "base_commit": task["base_commit"],
            "environment_setup_commit": task.get("environment_setup_commit", task["base_commit"]),
            "problem_statement": task.get("problem_statement", ""),
            "test_patch": task.get("test_patch", ""),
            "FAIL_TO_PASS": task.get("fail_to_pass", []),
            "PASS_TO_PASS": task.get("pass_to_pass", []),
        }
        spec = make_test_spec(instance)
        status_map = MAP_REPO_TO_PARSER[task["repo"]](output, spec)
        eval_type = EvalType.FAIL_ONLY if task["repo"] in FAIL_ONLY_REPOS else EvalType.PASS_AND_FAIL
        report = get_eval_tests_report(
            status_map,
            {FAIL_TO_PASS: spec.FAIL_TO_PASS, PASS_TO_PASS: spec.PASS_TO_PASS},
            eval_type=eval_type,
        )
        # The online-RL reward is dense on the target signal (passed
        # FAIL_TO_PASS tests divided by all relevant FAIL_TO_PASS tests), while
        # PASS_TO_PASS remains a multiplicative regression guard.  A repair
        # that breaks previously passing tests therefore cannot receive a
        # higher reward.  ``compute_fail_to_pass`` is binary, so it cannot
        # expose partial repair progress.
        f2p_success = len(report[FAIL_TO_PASS]["success"])
        f2p_total = len(spec.FAIL_TO_PASS)
        f2p = f2p_success / f2p_total if f2p_total else 0.0
        p2p = compute_pass_to_pass(report)
        resolved = get_resolution_status(report) == ResolvedStatus.FULL.value
        return RewardResult(
            reward=round(float(f2p * p2p), 6),
            resolved=resolved,
            fail_to_pass_success=f2p_success,
            fail_to_pass_total=f2p_total,
            pass_to_pass_success=len(report[PASS_TO_PASS]["success"]),
            pass_to_pass_total=len(spec.PASS_TO_PASS),
            test_exit_code=exit_code,
            parser_ok=bool(status_map),
            raw_tail=output[-4000:],
        )
    except Exception as exc:
        return RewardResult(
            reward=0.0,
            resolved=False,
            fail_to_pass_success=0,
            fail_to_pass_total=len(task.get("fail_to_pass", [])),
            pass_to_pass_success=0,
            pass_to_pass_total=len(task.get("pass_to_pass", [])),
            test_exit_code=exit_code,
            parser_ok=False,
            raw_tail=f"grader error: {exc}\n{output[-3500:]}",
        )
