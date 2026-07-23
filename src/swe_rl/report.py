"""Create an acceptance-friendly reward curve from VERL's JSONL file logger."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _metrics(record: dict[str, Any]) -> dict[str, Any]:
    data = record.get("data")
    return data if isinstance(data, dict) else record


def _extract_series(
    records: list[dict[str, Any]], key: str
) -> tuple[list[int], list[float]]:
    steps, values = [], []
    for record in records:
        payload = _metrics(record)
        value = payload.get(key)
        if value is None:
            continue
        step = record.get("global_step", record.get("step", len(steps)))
        if not isinstance(step, (int, float)):
            continue
        steps.append(int(step))
        values.append(float(value))
    return steps, values


def plot_reward(metrics_path: str | Path, output_path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    records: list[dict[str, Any]] = []
    with Path(metrics_path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))

    # ── collect available metric keys ──
    available = {
        key
        for record in records
        for key, value in _metrics(record).items()
        if isinstance(value, (int, float))
    }

    train_key = next(
        (
            k
            for k in ("critic/score/mean", "critic/rewards/mean", "actor/reward/mean", "reward/mean")
            if k in available
        ),
        "",
    )
    val_key = next(
        (
            k
            for k in (
                "val-core/swe_bench_verified/reward/mean@1",
                "val-core/swe_bench_verified/reward@1/mean",
                "validation/reward/mean",
            )
            if k in available
        ),
        "",
    )

    if not train_key and not val_key:
        raise ValueError("no numeric train or validation reward metric found")

    figure, axis = plt.subplots(figsize=(10, 5.5))

    # ── train reward (line) ──
    if train_key:
        t_steps, t_values = _extract_series(records, train_key)
        if t_steps:
            axis.plot(
                t_steps, t_values, color="#2c7be5", marker="o", markersize=3,
                linewidth=1.2, label=f"train ({train_key.rsplit('/', 1)[-1]})",
            )

    # ── val reward (scatter markers at val steps) ──
    if val_key:
        v_steps, v_values = _extract_series(records, val_key)
        if v_steps:
            axis.scatter(
                v_steps, v_values, color="#e63757", marker="D", s=60, zorder=5,
                label=f"val ({val_key.rsplit('/', 1)[-1]})",
            )

    axis.set(
        title="SWE-RL reward trajectory",
        xlabel="training step",
        ylabel="reward",
    )
    axis.legend(loc="upper left")
    axis.grid(alpha=0.25)
    figure.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output
