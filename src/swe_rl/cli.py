"""Single entrypoint; replaces the old 01..06 and 05-before script chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _json(data) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True))


def _prepare(args: argparse.Namespace) -> int:
    from swe_rl.data.prepare import prepare_verified

    _json(
        prepare_verified(
            output_dir=args.output,
            selected_count=args.count,
            eval_size=args.eval_size,
            seed=args.seed,
            dataset_name=args.dataset,
            exclude_repos=args.exclude_repos,
        )
    )
    return 0


def _validate_trace(args: argparse.Namespace) -> int:
    from swe_rl.schema import RewardResult, TraceEpisode, TraceStep

    valid = 0
    invalid = []
    with Path(args.path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                raw["steps"] = [TraceStep(**step) for step in raw["steps"]]
                raw["reward"] = RewardResult(**raw["reward"])
                episode = TraceEpisode(**raw)
                episode.validate(minimum_steps=args.minimum_steps)
                valid += 1
            except Exception as exc:
                invalid.append({"line": line_number, "error": str(exc)})
    _json({"valid": valid, "invalid": invalid, "ok": not invalid})
    return 0 if not invalid else 2


def _doctor(args: argparse.Namespace) -> int:
    from swe_rl.doctor import run_doctor

    result = run_doctor(require_secrets=args.require_secrets)
    _json(result)
    return 0 if result["ok"] else 2


def _evaluate(args: argparse.Namespace) -> int:
    from swe_rl.eval.runner import evaluate_dataset

    _json(
        evaluate_dataset(
            dataset_path=args.dataset,
            output_path=args.output,
            base_url=args.base_url,
            model=args.model,
            max_turns=args.max_turns,
            max_action_tokens=args.max_action_tokens,
            concurrency=args.concurrency,
            temperature=args.temperature,
        )
    )
    return 0


def _plot_metrics(args: argparse.Namespace) -> int:
    from swe_rl.report import plot_reward

    print(plot_reward(args.metrics, args.output))
    return 0



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="swe-rl", description="SWE repair RL pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="check dependencies and configuration")
    doctor.add_argument("--require-secrets", action="store_true")
    doctor.set_defaults(func=_doctor)

    data = sub.add_parser("prepare-data", help="create stratified prompt-only VERL parquet files")
    data.add_argument("--output", default="data/verified-edit")
    data.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    data.add_argument("--count", type=int, default=500)
    data.add_argument("--eval-size", type=int, default=50)
    data.add_argument("--seed", type=int, default=42)
    data.set_defaults(func=_prepare)

    trace = sub.add_parser("validate-trace", help="validate tracing JSONL")
    trace.add_argument("path")
    trace.add_argument("--minimum-steps", type=int, default=3)
    trace.set_defaults(func=_validate_trace)

    evaluate = sub.add_parser("evaluate", help="run held-out pass@1 through AGS")
    evaluate.add_argument("--dataset", default="data/verified-edit/eval.parquet")
    evaluate.add_argument("--output", default="outputs/eval.jsonl")
    evaluate.add_argument("--base-url", required=True)
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--max-turns", type=int, default=40)
    evaluate.add_argument("--max-action-tokens", type=int, default=3072)
    evaluate.add_argument("--concurrency", type=int, default=4)
    evaluate.add_argument("--temperature", type=float, default=0.0)
    evaluate.set_defaults(func=_evaluate)

    plot = sub.add_parser("plot-metrics", help="render a reward curve from VERL JSONL metrics")
    plot.add_argument("metrics")
    plot.add_argument("--output", default="outputs/reward-curve.png")
    plot.set_defaults(func=_plot_metrics)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
