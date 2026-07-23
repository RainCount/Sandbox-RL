"""On-policy VERL AgentLoop whose commands execute in Tencent AGS.

Token alignment follows VERL AgentLoop semantics with model-only policy masks.
The model, actions, observations and execution reward belong to one trajectory.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput, register

from swe_rl.agent.protocol import TOOLS, build_initial_messages, extract_action
from swe_rl.agent.policy import looks_like_environment_mutation, looks_like_focused_test
from swe_rl.agent.trajectory import TokenTrajectory, TrajectoryAlignmentError, transition_suffix
from swe_rl.config import AGSConfig, RunConfig, TCRConfig
from swe_rl.sandbox.ags import AGSSWEEnvironment
from swe_rl.schema import TraceEpisode, TraceStep
from swe_rl.verl.token_safety import sanitize_prompt_token_ids

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_PROCESS_SEMAPHORE: asyncio.Semaphore | None = None
_ASSISTANT_BOUNDARY = "__SWE_RL_ASSISTANT_BOUNDARY_4d3f08a1__"


def _generation_logit_bias() -> dict[int, float]:
    """Return a vLLM-native hard suppression for configured bad token IDs.

    This sits on each generation request, so it remains active after VERL
    replaces the actor LoRA adapter.  It is deliberately opt-in: model-specific
    tokenizer holes belong in the deployment environment, not in agent logic.
    """
    raw_ids = os.environ.get("SWE_RL_BANNED_TOKEN_IDS", "").strip()
    if not raw_ids:
        return {}
    try:
        return {
            int(token_id.strip()): -100.0
            for token_id in raw_ids.split(",")
            if token_id.strip()
        }
    except ValueError as exc:
        raise ValueError(
            "SWE_RL_BANNED_TOKEN_IDS must be a comma-separated list of integer IDs"
        ) from exc


def _log_generation_event(event: str, *, instance_id: str, sample_index: int, **fields: Any) -> None:
    """Emit rollout progress that is visible even while no sandbox command is running."""
    payload = {
        "event": event,
        "instance_id": instance_id,
        "sample_index": sample_index,
        **fields,
    }
    logger.info("agent_generation=%s", json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _semaphore() -> asyncio.Semaphore:
    global _PROCESS_SEMAPHORE
    if _PROCESS_SEMAPHORE is None:
        _PROCESS_SEMAPHORE = asyncio.Semaphore(int(os.environ.get("AGS_MAX_CONCURRENCY", "24")))
    return _PROCESS_SEMAPHORE


@register("ags_swe")
class AGSSWEAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        self.enable_thinking = bool(kwargs.pop("enable_thinking", False))
        self.max_turns = int(kwargs.pop("max_turns", 40))
        self.max_action_tokens = int(kwargs.pop("max_action_tokens", 3072))
        self.command_timeout = int(kwargs.pop("command_timeout", 180))
        self.test_timeout = int(kwargs.pop("test_timeout", 120))
        super().__init__(*args, **kwargs)
        self.apply_chat_template_kwargs = {
            **getattr(self, "apply_chat_template_kwargs", {}),
            "enable_thinking": self.enable_thinking,
        }
        self.generation_logit_bias = _generation_logit_bias()
        if self.generation_logit_bias:
            logger.warning(
                "Applying vLLM-native logit bias to generation token IDs: %s",
                sorted(self.generation_logit_bias),
            )

    async def _render(self, messages: list[dict[str, str]]) -> list[int]:
        return list(await self.apply_chat_template(messages, tools=TOOLS))

    def _render_transition(self, observation: str, generated_ids: list[int]) -> list[int]:
        """Render only the suffix between one assistant turn and the next.

        Some Qwen-family chat templates insert an empty thinking block in
        generation prompts, so a full conversation re-render is intentionally
        not used for prefix alignment.
        """
        messages = [
            {"role": "assistant", "content": _ASSISTANT_BOUNDARY},
            {"role": "tool", "content": observation},
        ]
        encoded = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            **getattr(self, "apply_chat_template_kwargs", {}),
        )
        if hasattr(encoded, "keys"):
            encoded = encoded["input_ids"]
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        rendered = list(encoded)
        marker_ids = list(self.tokenizer.encode(_ASSISTANT_BOUNDARY, add_special_tokens=False))
        return transition_suffix(rendered, marker_ids, generated_ids)

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        started = time.monotonic()
        task = kwargs.get("extra_info", {})
        if isinstance(task, str):
            task = json.loads(task)
        if not isinstance(task, dict) or not task.get("instance_id"):
            raise ValueError("extra_info with a SWE-bench instance_id is required")

        sample_index = int(kwargs.get("session_id", task.get("sample_index", 0)))
        messages = build_initial_messages(task)
        trace_steps: list[TraceStep] = []
        command_fingerprints: list[str] = []
        successful_source_edits = 0
        last_source_edit_turn: int | None = None
        consecutive_loop_refusals = 0
        generation_request_id = f"{task['instance_id']}:{sample_index}:{uuid4().hex}"
        environment: AGSSWEEnvironment | None = None
        patch = ""
        alignment_error = ""

        async with _semaphore():
            try:
                environment = await asyncio.to_thread(
                    AGSSWEEnvironment(task, AGSConfig.from_env(), TCRConfig.from_env()).__enter__
                )
                trajectory = TokenTrajectory(await self._render(messages))
                for turn_index in range(1, self.max_turns + 1):
                    prompt_ids, unsafe_token_ids = sanitize_prompt_token_ids(
                        self.tokenizer, trajectory.token_ids
                    )
                    if unsafe_token_ids:
                        logger.warning(
                            "Replacing tokenizer-hole prompt IDs before vLLM generation: "
                            "instance=%s sample=%s "
                            "turn=%s ids=%s",
                            task["instance_id"],
                            sample_index,
                            turn_index,
                            sorted(set(unsafe_token_ids)),
                        )
                    remaining = self._remaining_budget(len(prompt_ids), len(trajectory.response_mask))
                    if remaining < 64:
                        trace_steps.append(
                            TraceStep(
                                step=turn_index,
                                action="context_limit",
                                observation="Context budget exhausted",
                                done=True,
                            )
                        )
                        break
                    generation_started = time.monotonic()
                    _log_generation_event(
                        "started",
                        instance_id=task["instance_id"],
                        sample_index=sample_index,
                        turn=turn_index,
                        max_tokens=min(
                            int(sampling_params.get("max_tokens") or self.max_action_tokens),
                            self.max_action_tokens,
                            remaining,
                        ),
                        remaining_tokens=remaining,
                    )
                    output = await self.server_manager.generate(
                        request_id=generation_request_id,
                        prompt_ids=prompt_ids,
                        sampling_params={
                            **sampling_params,
                            # SERA tokenizer hole tokens (151669-151935) are banned
                            # via a sampler-level patch (vllm-v1-ban-hole-tokens.py)
                            # plus vLLM-native logit_bias as a fallback.
                            "logit_bias": {
                                **sampling_params.get("logit_bias", {}),
                                **self.generation_logit_bias,
                            },
                            "logprobs": sampling_params.get("logprobs", True),
                            "max_tokens": min(
                                int(sampling_params.get("max_tokens") or self.max_action_tokens),
                                self.max_action_tokens,
                                remaining,
                            ),
                        },
                    )
                    response_ids = list(output.token_ids)
                    response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                    raw_response_text = self.tokenizer.decode(response_ids, skip_special_tokens=False)
                    special_token_ids = set(getattr(self.tokenizer, "all_special_ids", []))
                    non_special_token_count = sum(
                        token_id not in special_token_ids for token_id in response_ids
                    )
                    _log_generation_event(
                        "finished",
                        instance_id=task["instance_id"],
                        sample_index=sample_index,
                        turn=turn_index,
                        duration_seconds=round(time.monotonic() - generation_started, 3),
                        output_tokens=len(response_ids),
                        preview=response_text[:96].replace("\n", "\\n"),
                        raw_preview=raw_response_text[:96].replace("\n", "\\n"),
                        non_special_token_count=non_special_token_count,
                        first_token_ids=response_ids[:12],
                    )
                    if response_ids and not response_text.strip():
                        logger.warning(
                            "vLLM returned no visible agent text: instance=%s sample=%s turn=%s "
                            "tokens=%s raw=%r",
                            task["instance_id"],
                            sample_index,
                            turn_index,
                            response_ids[:32],
                            raw_response_text[:256],
                        )
                    response_logprobs = (
                        list(output.log_probs) if output.log_probs is not None else [0.0] * len(response_ids)
                    )
                    trajectory.append_generation(response_ids, response_logprobs)
                    action, format_error = extract_action(response_text)
                    messages.append({"role": "assistant", "content": response_text})
                    if format_error:
                        _log_generation_event(
                            "format_error",
                            instance_id=task["instance_id"],
                            sample_index=sample_index,
                            turn=turn_index,
                            reason=format_error,
                        )
                        observation = (
                            format_error + ". Call exactly one provided tool with compact, valid arguments."
                        )
                        if "truncated tool call" in format_error:
                            observation += (
                                " Do not retry the same function-sized payload. "
                                "Use bash `sed -n` to read the "
                                "precise target and then edit a unique expression or 1-8 lines."
                            )
                        trace_steps.append(
                            TraceStep(
                                step=turn_index,
                                action="<format_error>",
                                observation=observation,
                                model_response=response_text,
                                return_code=2,
                            )
                        )
                        messages.append({"role": "user", "content": f"OBSERVATION:\n{observation}"})
                    else:
                        assert action is not None
                        fingerprint = " ".join(action.split())
                        is_edit = action.lstrip().startswith(("STR_REPLACE ", "STR_CREATE "))
                        # A second identical read-only command never supplies new evidence.
                        # Refuse it early rather than allowing a small model to consume the
                        # entire trajectory on the same grep/sed call.
                        if (
                            not is_edit
                            and not looks_like_focused_test(action)
                            and command_fingerprints[-6:].count(fingerprint) >= 1
                        ):
                            observation = (
                                "LOOP_REFUSED: this exact read-only action was already run. "
                                "Use its output: inspect a different location, or make a small "
                                "source edit. Do not repeat grep/sed."
                            )
                            result_code = 2
                            duration = 0.0
                            command_fingerprints.append(fingerprint)
                        elif action == "submit":
                            if not await asyncio.to_thread(environment.has_source_changes):
                                observation = (
                                    "SUBMIT_REFUSED: git diff is empty; make a source-code change first."
                                )
                                result_code = 2
                                duration = 0.0
                            else:
                                observation = "Submission accepted; running the official task tests."
                                trace_steps.append(
                                    TraceStep(
                                        step=turn_index,
                                        action=action,
                                        observation=observation,
                                        model_response=response_text,
                                        return_code=0,
                                        done=True,
                                    )
                                )
                                break
                        elif looks_like_environment_mutation(action):
                            observation = (
                                "ENVIRONMENT_MUTATION_REFUSED: package installation is outside the "
                                "coding task and the sandbox has no general package network. Work with "
                                "the dependencies already in the image; make a source fix or submit the "
                                "best existing patch."
                            )
                            result_code = 2
                            duration = 0.0
                            command_fingerprints.append(fingerprint)
                        else:
                            command_fingerprints.append(fingerprint)
                            command_result = await asyncio.to_thread(
                                environment.execute, action, timeout=self.command_timeout
                            )
                            observation = command_result.output or "Your command ran successfully and did not produce any output."
                            result_code = command_result.return_code
                            duration = command_result.duration_seconds
                            if is_edit and result_code != 0:
                                observation += (
                                    "\nEDIT_FAILED: the SEARCH text must be an exact unique copy "
                                    "from the file. Do not guess or retry this large block. Run one "
                                    "short `sed -n 'START,ENDp' path` for the intended 1-4 lines, "
                                    "then replace only those exact lines."
                                )
                        trace_steps.append(
                            TraceStep(
                                step=turn_index,
                                action=action,
                                observation=observation,
                                model_response=response_text,
                                return_code=result_code,
                                duration_seconds=duration,
                            )
                        )
                        if is_edit and result_code == 0:
                            successful_source_edits += 1
                            last_source_edit_turn = turn_index
                        if observation.startswith(("LOOP_REFUSED:", "ENVIRONMENT_MUTATION_REFUSED:")):
                            consecutive_loop_refusals += 1
                        else:
                            consecutive_loop_refusals = 0

                        should_finish = False
                        if (
                            successful_source_edits
                            and result_code == 0
                            and looks_like_focused_test(action)
                        ):
                            observation += (
                                "\nAUTO_SUBMIT: a focused test passed after a source edit; "
                                "preserving this candidate for official evaluation."
                            )
                            trace_steps[-1].observation = observation
                            trace_steps[-1].done = True
                            should_finish = True
                        elif consecutive_loop_refusals >= 3:
                            suffix = (
                                "preserving the current source patch for official evaluation."
                                if successful_source_edits
                                else "ending this zero-patch trajectory without wasting more rollout budget."
                            )
                            observation += "\nAUTO_STOP_STALLED: three refused actions supplied no new evidence; " + suffix
                            trace_steps[-1].observation = observation
                            trace_steps[-1].done = True
                            should_finish = True
                        elif (
                            last_source_edit_turn is not None
                            and turn_index - last_source_edit_turn >= 8
                        ):
                            observation += (
                                "\nAUTO_SUBMIT_EDIT_BUDGET: eight tool calls after the latest source edit "
                                "did not produce a better edit; preserving the candidate for official evaluation."
                            )
                            trace_steps[-1].observation = observation
                            trace_steps[-1].done = True
                            should_finish = True
                        elif not successful_source_edits and turn_index >= 24:
                            observation += (
                                "\nAUTO_STOP_NO_PATCH: no production edit was made within 24 tool calls; "
                                "ending this trajectory so the GRPO batch can continue."
                            )
                            trace_steps[-1].observation = observation
                            trace_steps[-1].done = True
                            should_finish = True
                        elif successful_source_edits and turn_index in {12, 18}:
                            observation += (
                                "\nBUDGET_STATUS: source changes exist. Run one focused test now; if it "
                                "passes, call submit. Do not resume broad search or repeat prior commands."
                            )
                            trace_steps[-1].observation = observation
                        messages.append(
                            {"role": "user", "content": f"OBSERVATION:\n{observation}"}
                        )
                        if result_code == 124 or should_finish:
                            trace_steps[-1].done = True
                            break

                    if turn_index < self.max_turns:
                        try:
                            transition = self._render_transition(messages[-1]["content"], response_ids)
                            available = self._remaining_budget(
                                len(trajectory.token_ids), len(trajectory.response_mask)
                            )
                            if not trajectory.append_observation(transition, limit=available):
                                trace_steps.append(
                                    TraceStep(
                                        step=turn_index + 1,
                                        action="context_limit",
                                        observation="Observation exhausted the trajectory token budget",
                                        done=True,
                                    )
                                )
                                break
                        except TrajectoryAlignmentError as exc:
                            alignment_error = str(exc)
                            trace_steps.append(
                                TraceStep(
                                    step=turn_index + 1,
                                    action="alignment_error",
                                    observation=alignment_error,
                                    done=True,
                                )
                            )
                            break

                if not trace_steps:
                    trace_steps.append(
                        TraceStep(step=1, action="startup", observation="No model turn completed", done=True)
                    )
                trace_steps[-1].done = True
                reward, _test_output = await asyncio.to_thread(
                    environment.run_tests, timeout=self.test_timeout
                )
                patch = await asyncio.to_thread(environment.patch)
                trace_steps[-1].reward = reward.reward
                result = self._build_output(
                    trajectory, trace_steps, reward, patch, task, started, alignment_error
                )
                self._persist_trace(
                    result, trace_steps, reward, patch, messages[:2], task, sample_index
                )
                return result
            finally:
                if environment is not None:
                    await asyncio.to_thread(environment.close)

    def _remaining_budget(self, prompt_length: int, response_length: int) -> int:
        """Calculate how many response tokens remain in the trajectory budget.

        rollout.response_length is the TOTAL response budget for the entire
        multi-turn trajectory (not per-turn). Per-turn cap is enforced by
        self.max_action_tokens in the generate() call.
        """
        rollout = getattr(self.config, "actor_rollout_ref", self.config).rollout
        max_model_len = int(getattr(rollout, "max_model_len", 0) or 0)
        max_response_length = int(
            getattr(rollout, "response_length", 0) or self.data_config.max_response_length
        )
        return max(
            0,
            min(
                max_response_length - response_length,
                max_model_len - prompt_length if max_model_len else max_response_length,
            ),
        )

    def _build_output(
        self, trajectory, steps, reward, patch, task, started, alignment_error
    ) -> AgentLoopOutput:
        rollout = getattr(self.config, "actor_rollout_ref", self.config).rollout
        maximum = int(rollout.response_length)
        pad = getattr(self.tokenizer, "pad_token_id", 0) or 0
        prompt_ids = trajectory.initial_prompt_ids or [pad]
        response_ids = list(trajectory.response_ids[:maximum]) or [pad]
        response_mask = list(trajectory.response_mask[:maximum]) or [1]
        if not any(response_mask):
            response_mask[-1] = 1
        logprobs = list(trajectory.response_logprobs[: len(response_ids)])
        logprobs.extend([0.0] * (len(response_ids) - len(logprobs)))
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=logprobs,
            reward_score=reward.reward if not alignment_error else 0.0,
            num_turns=sum(step.action not in {"context_limit", "alignment_error"} for step in steps),
            metrics=AgentLoopMetrics(
                generate_sequences=time.monotonic() - started,
                tool_calls=float(len(steps)),
                compute_score=0.0,
            ),
            extra_fields={
                "instance_id": task["instance_id"],
                "resolved": reward.resolved,
                "generated_patch": patch,
                "reward_details": asdict(reward),
                "alignment_failed": bool(alignment_error),
                "alignment_failure_reason": alignment_error,
            },
        )

    def _persist_trace(
        self, output, steps, reward, patch, prompt_messages, task, sample_index: int
    ) -> None:
        run = RunConfig.from_env()
        episode = TraceEpisode(
            instance_id=task["instance_id"],
            model=run.model_path,
            sample_index=sample_index,
            steps=steps,
            reward=reward,
            generated_patch=patch,
            prompt_messages=prompt_messages,
            response_token_count=len(output.response_ids),
            response_mask_count=sum(output.response_mask),
            run_id=run.run_id,
            round_id=run.round_id,
        )
        directory = Path(run.trace_dir) / "episodes"
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{episode.episode_id}.json"
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory, delete=False) as handle:
            handle.write(episode.to_json())
            temporary = Path(handle.name)
        temporary.replace(destination)
