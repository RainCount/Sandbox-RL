"""Patch vLLM V1 sampler to ban model-vocab hole tokens globally.

SERA tokenizer covers only IDs 0-151668 while the model embedding has 151936
rows. Tokens 151669-151935 are "holes" that must never be generated.

On vLLM 0.11.0, per-request logit_bias should work, but we keep this sampler-
level patch as a belt-and-suspenders measure: it bans the hole tokens directly
in the Sampler.forward() method, after apply_penalties and before sample(),
so both greedy and stochastic paths see hole_tokens = -inf.

Runs at image build time, idempotent (uses _swrl_final_ban marker).
"""
import sys

try:
    import vllm.v1.sample.sampler as _mod
except Exception as exc:  # pragma: no cover
    print(f"ERROR: cannot import vllm.v1.sample.sampler: {exc}", file=sys.stderr)
    sys.exit(1)

path = getattr(_mod, "__file__", None)
if not path:
    print("ERROR: cannot locate vllm sampler.py __file__", file=sys.stderr)
    sys.exit(1)

src = open(path, encoding="utf-8").read()

final_marker = "_swrl_final_ban"
if final_marker in src:
    print(f"final ban already patched: {path}")
    sys.exit(0)

# Anchor: right after apply_penalties, just before sample().
anchor = "        logits = self.apply_penalties(logits, sampling_metadata)\n"
if anchor not in src:
    print(f"ERROR: anchor (apply_penalties) not found in {path}", file=sys.stderr)
    sys.exit(1)

insert = anchor + (
    "        # SWE-RL: ban model-vocab hole tokens directly in the sampler,\n"
    "        # placed AFTER apply_penalties and BEFORE sample() so that both\n"
    "        # greedy (argmax) and stochastic (topk/topp) paths see 151935=-inf.\n"
    "        # Earlier placement after apply_logits_bias was overridden by\n"
    "        # apply_penalties returning a fresh tensor (temp>0 still emitted\n"
    "        # 151935). This is the only reliable suppression for SERA holes.\n"
    "        _swrl_final_ban = __import__('os').environ.get("
    "'SWE_RL_BANNED_TOKEN_IDS', '')\n"
    "        if _swrl_final_ban:\n"
    "            for _swrl_tid in _swrl_final_ban.split(','):\n"
    "                _swrl_tid = _swrl_tid.strip()\n"
    "                if _swrl_tid:\n"
    "                    logits[:, int(_swrl_tid)] = float('-inf')\n"
)

src = src.replace(anchor, insert, 1)
open(path, "w", encoding="utf-8").write(src)
print(f"patched (final ban, before sample): {path}")

import importlib

importlib.reload(_mod)
print("import OK after patch")
