"""One immutable namespace per run/round; completion markers are written last."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunLayout:
    prefix: str
    run_id: str
    round_id: int

    @property
    def root(self) -> str:
        pieces = [self.prefix.strip("/"), "runs", self.run_id, f"rounds/{self.round_id:03d}"]
        return "/".join(piece for piece in pieces if piece)

    def key(self, relative: str) -> str:
        return f"{self.root}/{relative.lstrip('/')}"

    @property
    def train_prompts(self) -> str:
        return self.key("datasets/train.parquet")

    @property
    def eval_prompts(self) -> str:
        return self.key("datasets/eval.parquet")

    @property
    def trace_prefix(self) -> str:
        return self.key("traces/episodes")

    @property
    def checkpoint_prefix(self) -> str:
        return self.key("checkpoints")

    @property
    def export_prefix(self) -> str:
        return self.key("exports/huggingface")

    def marker(self, stage: str) -> str:
        if stage not in {"prepared", "trained", "exported", "evaluated"}:
            raise ValueError(f"unknown stage: {stage}")
        return self.key(f"status/{stage}.json")
