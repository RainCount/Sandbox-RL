from __future__ import annotations

from swe_rl.verl.token_safety import sanitize_prompt_token_ids


class _Tokenizer:
    def encode(self, text, *, add_special_tokens=False):
        assert text == "\n"
        assert not add_special_tokens
        return [198]

    def convert_ids_to_tokens(self, token_id):
        return {1: "a", 2: "b", 198: "Ċ"}.get(token_id)


def test_replaces_a_tokenizer_hole_with_a_real_boundary_token():
    ids, unsafe = sanitize_prompt_token_ids(_Tokenizer(), [1, 151935, 2])

    assert ids == [1, 198, 2]
    assert unsafe == [151935]


def test_leaves_valid_prompt_tokens_unchanged():
    ids, unsafe = sanitize_prompt_token_ids(_Tokenizer(), [1, 2])

    assert ids == [1, 2]
    assert unsafe == []
