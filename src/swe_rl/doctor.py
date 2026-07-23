"""Fail-fast checks before spending AGS or GPU quota."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

from swe_rl.config import SECRET_ENV_NAMES, AGSConfig, COSConfig, TCRConfig


def run_doctor(*, require_secrets: bool = False) -> dict[str, Any]:
    packages = ["pyarrow", "datasets", "swebench", "e2b_code_interpreter", "qcloud_cos"]
    package_status = {name: importlib.util.find_spec(name) is not None for name in packages}
    ags = AGSConfig.from_env(require_secret=False)
    cos = COSConfig.from_env(require_secret=False)
    tcr = TCRConfig.from_env()
    checks: dict[str, Any] = {
        "python_packages": package_status,
        "ags": {"domain": ags.domain, "template": ags.template, "credential_present": bool(ags.api_key)},
        "tcr": {
            "registry": tcr.registry,
            "namespace": tcr.namespace,
            "credential_present": bool(tcr.password),
        },
        "cos": {"region": cos.region, "bucket": cos.bucket, "credential_present": bool(cos.secret_key)},
        "secrets_in_environment": {name: bool(os.environ.get(name)) for name in SECRET_ENV_NAMES},
        "cwd": str(Path.cwd()),
    }
    required_ok = all(package_status.values())
    if require_secrets:
        required_ok = required_ok and bool(ags.api_key and cos.bucket and cos.secret_id and cos.secret_key)
    checks["ok"] = required_ok
    return checks
