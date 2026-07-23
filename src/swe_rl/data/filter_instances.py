"""Remove verified-bad task instances from a staged VERL parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def filter_parquet(path: str | Path, excluded_ids: set[str]) -> dict[str, object]:
    parquet_path = Path(path)
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    kept = [
        row
        for row in rows
        if str((row.get("extra_info") or {}).get("instance_id", "")) not in excluded_ids
    ]
    removed = [
        str((row.get("extra_info") or {}).get("instance_id", ""))
        for row in rows
        if str((row.get("extra_info") or {}).get("instance_id", "")) in excluded_ids
    ]
    if removed:
        temporary = parquet_path.with_suffix(parquet_path.suffix + ".filtered")
        pq.write_table(pa.Table.from_pylist(kept, schema=table.schema), temporary, compression="zstd")
        temporary.replace(parquet_path)
    return {
        "path": str(parquet_path),
        "before": len(rows),
        "after": len(kept),
        "removed": removed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet")
    parser.add_argument("--instance-ids", required=True)
    args = parser.parse_args()
    excluded = {value.strip() for value in args.instance_ids.split(",") if value.strip()}
    print(json.dumps(filter_parquet(args.parquet, excluded), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
