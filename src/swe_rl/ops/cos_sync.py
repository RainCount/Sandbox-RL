"""COS prefix transfer commands; no shell-embedded credentials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from swe_rl.config import COSConfig
from swe_rl.storage.cos import COSStore


def download_prefix(store: COSStore, prefix: str, destination: str | Path) -> int:
    destination = Path(destination)
    marker = ""
    count = 0
    while True:
        response = store.client.list_objects(
            Bucket=store.bucket,
            Prefix=prefix.rstrip("/") + "/",
            Marker=marker,
            MaxKeys=1000,
        )
        for item in response.get("Contents", []):
            key = item["Key"]
            relative = key[len(prefix.rstrip("/") + "/") :]
            if not relative:
                continue
            target = destination / relative
            # The COS SDK downloads through a temporary file and atomically
            # renames it. A matching final size is therefore a safe reusable
            # node-cache hit; partial files keep their temporary suffix.
            expected_size = int(item.get("Size", -1))
            if target.is_file() and expected_size >= 0 and target.stat().st_size == expected_size:
                continue
            store.download_file(key, target)
            count += 1
        if str(response.get("IsTruncated", "false")).lower() != "true":
            break
        marker = response["NextMarker"]
    return count


def list_prefix(store: COSStore, prefix: str, max_keys: int = 100) -> list[dict[str, object]]:
    response = store.client.list_objects(
        Bucket=store.bucket,
        Prefix=prefix.strip("/"),
        MaxKeys=max_keys,
    )
    return [{"key": item["Key"], "size": int(item.get("Size", 0))} for item in response.get("Contents", [])]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download-prefix")
    download.add_argument("prefix")
    download.add_argument("destination")
    upload = sub.add_parser("upload-tree")
    upload.add_argument("source")
    upload.add_argument("prefix")
    upload.add_argument("--overwrite", action="store_true")
    listing = sub.add_parser("list-prefix")
    listing.add_argument("prefix")
    listing.add_argument("--max-keys", type=int, default=100)
    marker = sub.add_parser("put-marker")
    marker.add_argument("key")
    marker.add_argument("--payload", default="{}")
    args = parser.parse_args(argv)

    store = COSStore(COSConfig.from_env())
    if args.command == "download-prefix":
        result = {"downloaded": download_prefix(store, args.prefix, args.destination)}
    elif args.command == "upload-tree":
        uploaded = store.upload_tree(args.source, args.prefix, overwrite=args.overwrite)
        result = {"uploaded": len(uploaded), "objects": uploaded}
    elif args.command == "list-prefix":
        objects = list_prefix(store, args.prefix, max_keys=args.max_keys)
        result = {"count": len(objects), "objects": objects}
    else:
        payload = json.loads(args.payload)
        store.put_marker(args.key, payload)
        result = {"marker": args.key}
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
