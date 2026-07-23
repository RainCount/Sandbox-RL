"""Make VERL's FSDP ``hf_model`` checkpoint contain merged LoRA weights.

VERL's ordinary FSDP Hugging Face branch gathers ``model.state_dict()``.
For a PEFT actor that state contains adapter wrapper keys, while the freshly
created Hugging Face base model expects ordinary model keys.  VERL already
ships and tests ``collect_merged_lora_params`` for FSDP1/FSDP2 rollout weight
sync.  Reuse it for deployable checkpoints, while retaining the normal gather
path for non-LoRA actors.
"""

from __future__ import annotations

from pathlib import Path


path = Path("/opt/verl/verl/utils/checkpoint/fsdp_checkpoint_manager.py")
source = path.read_text(encoding="utf-8")

import_before = (
    "from verl.utils.fsdp_utils import fsdp_version, "
    "get_fsdp_full_state_dict, get_fsdp_state_ctx"
)
import_after = (
    "from verl.utils.fsdp_utils import (\n"
    "    collect_merged_lora_params,\n"
    "    fsdp_version,\n"
    "    get_fsdp_full_state_dict,\n"
    "    get_fsdp_state_ctx,\n"
    ")"
)
state_before = (
    "            state_dict = get_fsdp_full_state_dict("
    "self.model, offload_to_cpu=True, rank0_only=True)"
)
state_after = (
    "            if self._has_lora():\n"
    "                # Export base-model keys with the trained adapters merged.\n"
    "                # The helper restores the live unmerged training weights.\n"
    "                state_dict = collect_merged_lora_params(self.model)\n"
    "            else:\n"
    "                state_dict = get_fsdp_full_state_dict(\n"
    "                    self.model, offload_to_cpu=True, rank0_only=True\n"
    "                )"
)

if "state_dict = collect_merged_lora_params(self.model)" in source:
    print(f"already patched: {path}")
    raise SystemExit(0)

if source.count(import_before) != 1:
    raise RuntimeError("unexpected VERL fsdp_utils import; refusing an unsafe patch")
if source.count(state_before) != 1:
    raise RuntimeError("unexpected VERL hf_model checkpoint branch; refusing an unsafe patch")

source = source.replace(import_before, import_after).replace(state_before, state_after)
compile(source, str(path), "exec")
path.write_text(source, encoding="utf-8")
print(f"patched merged LoRA Hugging Face checkpoint export: {path}")
