"""Evaluate an exported OpenAI-compatible model against held-out AGS tasks."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from openai import OpenAI

from swe_rl.agent.protocol import TOOLS, build_initial_messages, extract_action
from swe_rl.config import AGSConfig, RunConfig, TCRConfig
from swe_rl.sandbox.ags import AGSSWEEnvironment
from swe_rl.schema import TraceEpisode, TraceStep


def _task_rows(dataset_path: str | Path) -> list[dict[str, Any]]:
    table = pq.read_table(dataset_path)
    return [dict(row["extra_info"]) for row in table.to_pylist()]


def evaluate_one(
    task: dict[str, Any],
    *,
    base_url: str,
    model: str,
    max_turns: int,
    max_action_tokens: int,
    temperature: float,
) -> TraceEpisode:
    client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=180, max_retries=2)
    messages = build_initial_messages(task)
    initial_messages = [dict(message) for message in messages]
    steps: list[TraceStep] = []
    fingerprints: list[str] = []
    with AGSSWEEnvironment(task, AGSConfig.from_env(), TCRConfig.from_env()) as environment:
        for index in range(1, max_turns + 1):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                temperature=temperature,
                top_p=0.95,
                max_tokens=max_action_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls or []
            if tool_calls:
                call = tool_calls[0]
                text = "<tool_call>" + json.dumps(
                    {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    }
                ) + "</tool_call>"
                messages.append(response_message.model_dump(exclude_none=True))
            else:
                text = response_message.content or ""
                messages.append({"role": "assistant", "content": text})
            action, error = extract_action(text)
            if error:
                observation, code, duration = error, 2, 0.0
                action = "<format_error>"
            elif action == "submit":
                if environment.has_source_changes():
                    steps.append(TraceStep(index, action, "Submission accepted", done=True, return_code=0))
                    break
                observation, code, duration = "SUBMIT_REFUSED: git diff is empty", 2, 0.0
            else:
                assert action is not None
                fingerprint = " ".join(action.split())
                if fingerprints[-6:].count(fingerprint) >= 2:
                    observation, code, duration = "LOOP_REFUSED: choose a different action", 2, 0.0
                else:
                    fingerprints.append(fingerprint)
                    result = environment.execute(action)
                    observation, code, duration = result.output, result.return_code, result.duration_seconds
            steps.append(
                TraceStep(index, action or "<none>", observation, return_code=code, duration_seconds=duration)
            )
            if tool_calls:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_calls[0].id,
                        "content": f"OBSERVATION (exit={code}):\n{observation}",
                    }
                )
            else:
                messages.append(
                    {"role": "user", "content": f"OBSERVATION (exit={code}):\n{observation}"}
                )
        if not steps:
            steps.append(TraceStep(1, "startup", "No model output", done=True))
        steps[-1].done = True
        reward, _ = environment.run_tests()
        patch = environment.patch()
        steps[-1].reward = reward.reward

    run = RunConfig.from_env()
    return TraceEpisode(
        instance_id=task["instance_id"],
        model=model,
        sample_index=0,
        steps=steps,
        reward=reward,
        generated_patch=patch,
        prompt_messages=initial_messages,
        response_token_count=0,
        response_mask_count=0,
        run_id=run.run_id,
        round_id=run.round_id,
    )


def evaluate_dataset(
    *,
    dataset_path: str | Path,
    output_path: str | Path,
    base_url: str,
    model: str,
    max_turns: int = 40,
    max_action_tokens: int = 3072,
    concurrency: int = 4,
    temperature: float = 0.0,
) -> dict[str, Any]:
    started = time.monotonic()
    tasks = _task_rows(dataset_path)
    episodes: list[TraceEpisode] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                evaluate_one,
                task,
                base_url=base_url,
                model=model,
                max_turns=max_turns,
                max_action_tokens=max_action_tokens,
                temperature=temperature,
            ): task["instance_id"]
            for task in tasks
        }
        for future in as_completed(futures):
            episodes.append(future.result())
    episodes.sort(key=lambda episode: episode.instance_id)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(episode.to_json() + "\n")
    resolved = sum(episode.reward.resolved for episode in episodes)
    return {
        "model": model,
        "episodes": len(episodes),
        "resolved": resolved,
        "pass_at_1": resolved / len(episodes) if episodes else 0.0,
        "mean_reward": sum(episode.reward.reward for episode in episodes) / len(episodes)
        if episodes
        else 0.0,
        "output": str(output),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
