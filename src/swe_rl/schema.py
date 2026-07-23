"""Stable JSON contracts shared by rollout, training and evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

SCHEMA_VERSION = "2.0"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TraceStep:
    step: int
    action: str
    observation: str
    model_response: str = ""
    reward: float = 0.0
    done: bool = False
    return_code: int | None = None
    started_at: str = field(default_factory=utc_now)
    duration_seconds: float = 0.0


@dataclass
class RewardResult:
    reward: float
    resolved: bool
    fail_to_pass_success: int
    fail_to_pass_total: int
    pass_to_pass_success: int
    pass_to_pass_total: int
    test_exit_code: int
    parser_ok: bool
    raw_tail: str = ""


@dataclass
class TraceEpisode:
    instance_id: str
    model: str
    sample_index: int
    steps: list[TraceStep]
    reward: RewardResult
    generated_patch: str
    prompt_messages: list[dict[str, str]]
    response_token_count: int
    response_mask_count: int
    run_id: str
    round_id: int
    episode_id: str = field(default_factory=lambda: uuid4().hex)
    schema_version: str = SCHEMA_VERSION
    created_at: str = field(default_factory=utc_now)

    def validate(self, *, minimum_steps: int = 3) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported trace schema: {self.schema_version}")
        if not self.instance_id:
            raise ValueError("instance_id is required")
        if len(self.steps) < minimum_steps:
            raise ValueError(f"trace needs at least {minimum_steps} steps")
        if not 0.0 <= self.reward.reward <= 1.0:
            raise ValueError("reward must be in [0, 1]")
        if self.response_mask_count > self.response_token_count:
            raise ValueError("response_mask_count cannot exceed response_token_count")
        if self.steps and not self.steps[-1].done:
            raise ValueError("last step must have done=true")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class RoundManifest:
    run_id: str
    round_id: int
    model: str
    train_dataset_key: str
    eval_dataset_key: str
    base_model_key: str
    parent_checkpoint_key: str = ""
    schema_version: str = SCHEMA_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=True)
