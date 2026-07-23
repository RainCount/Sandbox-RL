"""Typed, environment-only runtime configuration.

Secrets intentionally have no YAML fallback: this prevents a convenient local
config file from silently becoming a credential dump committed to Git.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _int_env(name: str, default: int) -> int:
    value = _env(name)
    return int(value) if value else default


def _tuple_env(name: str, default: str = "") -> tuple[str, ...]:
    return tuple(item.strip().rstrip("/") for item in _env(name, default).split(",") if item.strip())


@dataclass(frozen=True)
class AGSConfig:
    domain: str
    api_key: str
    proxy_url: str
    template: str
    timeout_seconds: int
    max_concurrency: int
    dockerhub_mirror: str
    dockerhub_fallback_prefixes: tuple[str, ...]

    @classmethod
    def from_env(cls, *, require_secret: bool = True) -> AGSConfig:
        config = cls(
            domain=_env("E2B_DOMAIN", "ap-shanghai.tencentags.com"),
            api_key=_env("E2B_API_KEY"),
            proxy_url=_env("AGS_PROXY_URL"),
            template=_env("AGS_TEMPLATE", "swe-rl-sandbox"),
            timeout_seconds=_int_env("AGS_TIMEOUT_SECONDS", 1800),
            max_concurrency=_int_env("AGS_MAX_CONCURRENCY", 24),
            dockerhub_mirror=_env("DOCKERHUB_MIRROR", "https://mirror.ccs.tencentyun.com"),
            dockerhub_fallback_prefixes=_tuple_env("DOCKERHUB_FALLBACK_PREFIXES", "docker.1ms.run"),
        )
        if require_secret and not config.api_key:
            raise ValueError("E2B_API_KEY is required")
        return config


@dataclass(frozen=True)
class TCRConfig:
    registry: str
    namespace: str
    username: str
    password: str

    @classmethod
    def from_env(cls) -> TCRConfig:
        return cls(
            registry=_env("TCR_REGISTRY"),
            namespace=_env("TCR_NAMESPACE", "swe-rl"),
            username=_env("TCR_USERNAME"),
            password=_env("TCR_PASSWORD"),
        )


@dataclass(frozen=True)
class COSConfig:
    region: str
    bucket: str
    prefix: str
    secret_id: str
    secret_key: str
    token: str

    @classmethod
    def from_env(cls, *, require_secret: bool = True) -> COSConfig:
        config = cls(
            region=_env("COS_REGION", "ap-shanghai"),
            bucket=_env("COS_BUCKET"),
            prefix=_env("COS_PREFIX", "swe-rl").strip("/"),
            secret_id=_env("COS_SECRET_ID"),
            secret_key=_env("COS_SECRET_KEY"),
            token=_env("COS_TOKEN"),
        )
        if require_secret and (not config.bucket or not config.secret_id or not config.secret_key):
            raise ValueError("COS_BUCKET, COS_SECRET_ID and COS_SECRET_KEY are required")
        return config


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    round_id: int
    trace_dir: str
    model_path: str

    @classmethod
    def from_env(cls) -> RunConfig:
        return cls(
            run_id=_env("RUN_ID", "local-dev"),
            round_id=_int_env("ROUND_ID", 0),
            trace_dir=_env("TRACE_DIR", "runtime/traces"),
            model_path=_env("MODEL_PATH", "/workspace/runtime/model"),
        )


SECRET_ENV_NAMES = (
    "E2B_API_KEY",
    "TCR_PASSWORD",
    "COS_SECRET_ID",
    "COS_SECRET_KEY",
    "COS_TOKEN",
    "WANDB_API_KEY",
)
