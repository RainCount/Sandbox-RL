"""Release the actor's CUDA cache before waking colocated vLLM weights.

VERL's v1 engine worker currently clears the PyTorch allocator only after the
rollout weights have been restored. Long-sequence actor updates can leave a
large reserved cache, so vLLM's cuMem allocator cannot remap the model weights.
This patch is deliberately strict: an upstream source change fails the image
build instead of silently leaving the OOM unfixed.
"""

from __future__ import annotations

import sys
from pathlib import Path


DEFAULT_TARGET = Path("/opt/verl/verl/workers/engine_workers.py")
NEEDLE = """        set_expandable_segments(False)
        log_gpu_memory_usage(\"Before resume weights\", logger=logger)

        # 1. resume rollout memory (weights were released during sleep)
"""
REPLACEMENT = """        set_expandable_segments(False)

        # The actor update and FSDP checkpoint path can leave tens of GiB in
        # PyTorch's reserved allocator. vLLM uses a separate cuMem allocator,
        # so release the inactive actor cache before remapping rollout weights.
        aggressive_empty_cache(force_sync=True)
        log_gpu_memory_usage(\"Before resume weights\", logger=logger)

        # 1. resume rollout memory (weights were released during sleep)
"""


def patch(target: Path) -> str:
    source = target.read_text(encoding="utf-8")
    if REPLACEMENT in source:
        return "already patched"
    count = source.count(NEEDLE)
    if count != 1:
        raise RuntimeError(f"expected exactly one VERL wake-up site, found {count}: {target}")
    target.write_text(source.replace(NEEDLE, REPLACEMENT), encoding="utf-8")
    return "patched"


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TARGET
    print(f"{patch(path)}: {path}")
