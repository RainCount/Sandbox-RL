"""Small COS client with immutable-by-default uploads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from swe_rl.config import COSConfig


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class COSStore:
    def __init__(self, config: COSConfig):
        from qcloud_cos import CosConfig, CosS3Client

        sdk_config = CosConfig(
            Region=config.region,
            SecretId=config.secret_id,
            SecretKey=config.secret_key,
            Token=config.token or None,
        )
        self.client = CosS3Client(sdk_config)
        self.bucket = config.bucket

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception as exc:
            status_getter = getattr(exc, "get_status_code", None)
            status = status_getter() if callable(status_getter) else None
            if str(status) == "404":
                return False
            raise

    def upload_file(self, source: str | Path, key: str, *, overwrite: bool = False) -> dict[str, Any]:
        source = Path(source)
        if not overwrite and self.exists(key):
            raise FileExistsError(f"COS object already exists: {key}")
        self.client.upload_file(Bucket=self.bucket, Key=key, LocalFilePath=str(source))
        return {"key": key, "size": source.stat().st_size, "sha256": sha256_file(source)}

    def download_file(self, key: str, destination: str | Path) -> Path:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(Bucket=self.bucket, Key=key, DestFilePath=str(destination))
        return destination

    def put_marker(self, key: str, payload: dict[str, Any], *, overwrite: bool = False) -> None:
        if not overwrite and self.exists(key):
            raise FileExistsError(f"COS marker already exists: {key}")
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
        self.client.put_object(Bucket=self.bucket, Key=key, Body=body, ContentType="application/json")

    def upload_tree(
        self, source_dir: str | Path, prefix: str, *, overwrite: bool = False
    ) -> list[dict[str, Any]]:
        source_dir = Path(source_dir)
        uploaded = []
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                key = f"{prefix.rstrip('/')}/{path.relative_to(source_dir).as_posix()}"
                uploaded.append(self.upload_file(path, key, overwrite=overwrite))
        return uploaded
