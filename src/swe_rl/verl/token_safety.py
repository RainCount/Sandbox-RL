"""Guardrails for tokenizer/model vocabulary mismatches during async rollouts."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


class UnsafePromptTokenError(ValueError):
    """Raised when an invalid prompt token cannot be replaced safely."""


def sanitize_prompt_token_ids(tokenizer: Any, token_ids: Iterable[int]) -> tuple[list[int], list[int]]:
    """Replace tokenizer holes with a newline token before passing IDs to vLLM.

    Some Qwen-family checkpoints declare a larger model vocabulary than their
    distributed tokenizer artifact.  Hugging Face can emit a reserved (hole)
    ID from a chat template while vLLM correctly rejects that ID.  A newline is
    semantically harmless at a message boundary and, unlike a pad token, is a
    real vocabulary token.
    """
    ids = [int(token_id) for token_id in token_ids]
    replacement = list(tokenizer.encode("\n", add_special_tokens=False))
    if not replacement:
        raise UnsafePromptTokenError("Tokenizer cannot encode the safe newline replacement token")

    unsafe: list[int] = []
    sanitized: list[int] = []
    for token_id in ids:
        try:
            token = tokenizer.convert_ids_to_tokens(token_id)
        except (KeyError, ValueError, TypeError):
            token = None
        if token is None:
            unsafe.append(token_id)
            sanitized.extend(replacement)
        else:
            sanitized.append(token_id)
    return sanitized, unsafe
