"""Promote the newest explicitly saved Hugging Face checkpoint for serving."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


def _step_number(path: Path) -> int:
    match = re.search(r"global_step_(\d+)", path.as_posix())
    return int(match.group(1)) if match else -1


def _is_complete_export(source: Path) -> bool:
    """Check whether a huggingface export directory has the necessary files."""
    if not (source / "config.json").exists():
        return False
    has_safetensors = bool(list(source.glob("*.safetensors")))
    has_adapter = bool(list(source.glob("adapter_model.*")))
    has_lora_meta = (source.parent / "lora_train_meta.json").exists()
    # LoRA checkpoints store adapter weights, not full safetensors
    if has_lora_meta and (has_adapter or has_safetensors):
        return True
    # Full FT checkpoints must have safetensors
    if has_safetensors:
        return True
    return False


def export_latest(checkpoint_root: str | Path, destination: str | Path) -> Path:
    root = Path(checkpoint_root)
    candidates = sorted(root.glob("global_step_*/actor/huggingface"), key=_step_number)
    if not candidates:
        raise FileNotFoundError(
            "No actor/huggingface checkpoint found. Ensure actor.checkpoint.save_contents "
            "includes 'hf_model'."
        )
    # Try candidates from newest to oldest
    source = None
    for candidate in reversed(candidates):
        if _is_complete_export(candidate):
            source = candidate
            break
    if source is None:
        # Fallback: accept any candidate and log the issue
        source = candidates[-1]
        print(f"Warning: No complete huggingface export found. Using {source} as fallback.", flush=True)
    destination = Path(destination)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    manifest = {
        "source": str(source),
        "global_step": _step_number(source),
        "format": "huggingface",
    }
    (destination / "swe_rl_export.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_root")
    parser.add_argument("destination")
    args = parser.parse_args(argv)
    print(export_latest(args.checkpoint_root, args.destination))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
