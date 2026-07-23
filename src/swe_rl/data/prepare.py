"""Build prompt-only VERL datasets with deterministic repository stratification."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from swe_rl.agent.protocol import build_initial_messages


def _stable_rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _list_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parsed = json.loads(value)
        return [str(item) for item in parsed]
    return [str(item) for item in value]


def repository_stratified_select(
    rows: Iterable[dict[str, Any]], limit: int, seed: int
) -> list[dict[str, Any]]:
    """Round-robin repositories after deterministic within-repository shuffling.

    This prevents `dataset[:N]` from producing a single-repository benchmark.
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["repo"])].append(dict(row))
    queues = {
        repo: deque(sorted(items, key=lambda item: _stable_rank(seed, str(item["instance_id"]))))
        for repo, items in groups.items()
    }
    repo_order = sorted(queues, key=lambda repo: _stable_rank(seed, repo))
    selected: list[dict[str, Any]] = []
    while len(selected) < limit and any(queues.values()):
        for repo in repo_order:
            if queues[repo] and len(selected) < limit:
                selected.append(queues[repo].popleft())
    return selected


def split_train_eval(
    rows: Iterable[dict[str, Any]], *, eval_size: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Choose a repository-balanced held-out set with no instance overlap."""
    rows = list(rows)
    eval_rows = repository_stratified_select(rows, min(eval_size, len(rows)), seed + 1)
    eval_ids = {str(row["instance_id"]) for row in eval_rows}
    train_rows = [row for row in rows if str(row["instance_id"]) not in eval_ids]
    if eval_ids & {str(row["instance_id"]) for row in train_rows}:
        raise AssertionError("train/eval leakage")
    return train_rows, eval_rows


def _test_command(row: dict[str, Any]) -> str:
    from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
    from swebench.harness.test_spec.python import get_test_directives

    command = MAP_REPO_VERSION_TO_SPECS[row["repo"]][row["version"]]["test_cmd"]
    if isinstance(command, list):
        command = command[-1]
    directives = get_test_directives(row)
    return " ".join([str(command), *directives])


def _docker_image(row: dict[str, Any]) -> str:
    from swebench.harness.test_spec.test_spec import make_test_spec

    # The remote image key also encodes "__" as "_1776_". Delegating this to
    # SWE-bench avoids generating plausible-looking but nonexistent image names.
    return str(make_test_spec(row, namespace="swebench").instance_image_key)


def to_verl_row(row: dict[str, Any], *, index: int, split: str) -> dict[str, Any]:
    instance_id = str(row["instance_id"])
    fail_to_pass = _list_value(row.get("FAIL_TO_PASS"))
    pass_to_pass = _list_value(row.get("PASS_TO_PASS"))
    problem = str(row["problem_statement"])
    extra_info = {
        "index": index,
        "split": split,
        "instance_id": instance_id,
        "repo": str(row["repo"]),
        "version": str(row.get("version", "")),
        "base_commit": str(row["base_commit"]),
        "environment_setup_commit": str(row.get("environment_setup_commit", row["base_commit"])),
        "problem_statement": problem,
        "test_patch": str(row.get("test_patch", "")),
        "fail_to_pass": fail_to_pass,
        "pass_to_pass": pass_to_pass,
        "test_command": _test_command(row),
        "docker_image": _docker_image(row),
    }
    return {
        "data_source": "swe_bench_verified",
        "prompt": build_initial_messages(extra_info),
        "ability": "code_repair",
        "reward_model": {"style": "rule", "ground_truth": instance_id},
        "agent_name": "ags_swe",
        "extra_info": extra_info,
    }


def write_parquet(rows: Iterable[dict[str, Any]], path: str | Path, *, split: str) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = [to_verl_row(row, index=index, split=split) for index, row in enumerate(rows)]
    pq.write_table(pa.Table.from_pylist(records), output, compression="zstd")
    return len(records)


def prepare_verified(
    *,
    output_dir: str | Path,
    selected_count: int = 500,
    eval_size: int = 50,
    seed: int = 42,
    dataset_name: str = "princeton-nlp/SWE-bench_Verified",
    dataset_split: str = "test",
    exclude_repos: list[str] | None = None,
) -> dict[str, Any]:
    """Prepare a repository-stratified SWE-bench Verified dataset.

    Args:
        exclude_repos: Repository names to exclude (e.g. ["psf/requests"]).
            psf/requests is excluded by default because its pytest suite makes
            network requests that hang indefinitely in the sandbox (no network),
            causing 900s timeouts and zero reward on every instance.
    """
    if exclude_repos is None:
        exclude_repos = ["psf/requests"]  # network-dependent pytest → always timeouts
    dataset_path = Path(dataset_name)
    if dataset_path.exists():
        if dataset_path.is_dir():
            parquet_files = sorted(dataset_path.glob(f"**/{dataset_split}-*.parquet"))
        else:
            parquet_files = [dataset_path]
        if not parquet_files:
            raise FileNotFoundError(f"no {dataset_split} parquet files found in {dataset_path}")
        dataset: Iterable[dict[str, Any]] = []
        local_rows: list[dict[str, Any]] = []
        for parquet_file in parquet_files:
            local_rows.extend(pq.read_table(parquet_file).to_pylist())
        dataset = local_rows
    else:
        if dataset_name.startswith(("runtime/", "./", "../")):
            raise FileNotFoundError(f"local dataset does not exist: {dataset_name}")
        from datasets import load_dataset

        dataset = load_dataset(dataset_name, split=dataset_split)
    # Filter out repos known to timeout in sandbox (no network access)
    filtered_rows = [
        row for row in dataset
        if str(row.get("repo", "")) not in exclude_repos
    ]
    num_excluded = len(list(dataset)) - len(filtered_rows)
    if num_excluded > 0:
        import logging
        _log = logging.getLogger(__name__)
        _log.info("Excluded %d instances from %s (timeout-prone in sandbox)", num_excluded, exclude_repos)
    selected = repository_stratified_select(filtered_rows, selected_count, seed)
    train_rows, eval_rows = split_train_eval(selected, eval_size=eval_size, seed=seed)
    output = Path(output_dir)
    train_count = write_parquet(train_rows, output / "train.parquet", split="train")
    eval_count = write_parquet(eval_rows, output / "eval.parquet", split="eval")
    manifest = {
        "schema_version": "2.0",
        "dataset": dataset_name,
        "source_split": dataset_split,
        "seed": seed,
        "train_count": train_count,
        "eval_count": eval_count,
        "train_instance_ids": [str(row["instance_id"]) for row in train_rows],
        "eval_instance_ids": [str(row["instance_id"]) for row in eval_rows],
        "repositories": sorted({str(row["repo"]) for row in selected}),
        "docker_image_namespace": "swebench",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
