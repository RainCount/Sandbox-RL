"""Pinned controller-side assets used to adapt the existing AGS template."""

from __future__ import annotations

import hashlib
import os
import shlex
import urllib.request
from pathlib import Path
from typing import Any

FUSE_OVERLAYFS_VERSION = "v1.14"
FUSE_OVERLAYFS_SHA256 = "4817a8896a9e6f0433080f88f5b71dec931e8829a89d64c71af94b0630ccb4a9"
FUSE_OVERLAYFS_URL = (
    "https://github.com/containers/fuse-overlayfs/releases/download/"
    f"{FUSE_OVERLAYFS_VERSION}/fuse-overlayfs-x86_64"
)
DEFAULT_FUSE_OVERLAYFS_BINARY = "/opt/swe-rl-assets/fuse-overlayfs"
STOP_DOCKERD_COMMAND = (
    "sudo pkill -TERM -x dockerd 2>/dev/null || true; "
    "for _attempt in $(seq 1 10); do pgrep -x dockerd >/dev/null || break; sleep 1; done; "
    "sudo pkill -KILL -x dockerd 2>/dev/null || true; sleep 1; "
    "sudo rm -f /var/run/docker.pid"
)

_FUSE_BINARY_CACHE: bytes | None = None


def dockerd_command(dockerhub_mirror: str) -> str:
    mirror = shlex.quote(dockerhub_mirror)
    return (
        "sudo nohup dockerd --storage-driver=fuse-overlayfs --iptables=false "
        "--ip6tables=false --userland-proxy=false "
        f"--registry-mirror={mirror} >/tmp/dockerd.log 2>&1 </dev/null &"
    )


def _command_output(result: Any) -> str:
    return (getattr(result, "stdout", "") or "") + (getattr(result, "stderr", "") or "")


def _verified_fuse_binary() -> bytes:
    global _FUSE_BINARY_CACHE
    if _FUSE_BINARY_CACHE is not None:
        return _FUSE_BINARY_CACHE
    path = Path(os.environ.get("FUSE_OVERLAYFS_BINARY", DEFAULT_FUSE_OVERLAYFS_BINARY))
    if path.is_file():
        payload = path.read_bytes()
    else:
        request = urllib.request.Request(FUSE_OVERLAYFS_URL, headers={"User-Agent": "swe-rl/2.0"})
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != FUSE_OVERLAYFS_SHA256:
        raise RuntimeError(f"fuse-overlayfs SHA256 mismatch: {digest}")
    _FUSE_BINARY_CACHE = payload
    return payload


def ensure_fuse_overlayfs(sandbox: Any) -> tuple[bool, bool]:
    """Return (available, installed_by_controller) without shell network access."""
    probe = sandbox.commands.run(
        "command -v fuse-overlayfs >/dev/null && echo READY || echo MISSING", timeout=10
    )
    if "READY" in _command_output(probe):
        return True, False
    sandbox.files.write("/tmp/fuse-overlayfs", _verified_fuse_binary())
    installed = sandbox.commands.run(
        "sudo install -m 0755 /tmp/fuse-overlayfs /usr/local/bin/fuse-overlayfs "
        "&& command -v fuse-overlayfs >/dev/null && echo READY || echo MISSING",
        timeout=30,
    )
    return "READY" in _command_output(installed), True
