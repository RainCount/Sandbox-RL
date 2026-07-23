"""Token-accurate assembly for multi-turn agent rollouts.

The initial prompt stays fixed. Model tokens and environment observations are
then appended to one response sequence; only model tokens participate in the
policy loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class TrajectoryAlignmentError(ValueError):
    """Raised when a chat-template transition cannot be located safely."""


def transition_suffix(
    rendered_ids: list[int], marker_ids: list[int], generated_ids: list[int]
) -> list[int]:
    """Return the template suffix after one unique assistant marker."""
    positions = [
        index
        for index in range(len(rendered_ids) - len(marker_ids) + 1)
        if rendered_ids[index : index + len(marker_ids)] == marker_ids
    ]
    if len(positions) != 1:
        raise TrajectoryAlignmentError(
            f"assistant boundary appears {len(positions)} times in the rendered transition"
        )
    suffix = rendered_ids[positions[0] + len(marker_ids) :]
    overlap = next(
        (
            size
            for size in range(min(len(generated_ids), len(suffix)), 0, -1)
            if generated_ids[-size:] == suffix[:size]
        ),
        0,
    )
    return suffix[overlap:]


@dataclass
class TokenTrajectory:
    initial_prompt_ids: list[int]
    token_ids: list[int] = field(init=False)
    response_mask: list[int] = field(default_factory=list)
    response_logprobs: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.initial_prompt_ids = list(self.initial_prompt_ids)
        self.token_ids = list(self.initial_prompt_ids)

    @property
    def response_ids(self) -> list[int]:
        return self.token_ids[len(self.initial_prompt_ids) :]

    def append_generation(self, token_ids: list[int], logprobs: list[float] | None) -> None:
        generated = list(token_ids)
        probabilities = list(logprobs) if logprobs is not None else [0.0] * len(generated)
        if len(probabilities) != len(generated):
            raise ValueError(
                f"generation token/logprob length mismatch: {len(generated)} != {len(probabilities)}"
            )
        self.token_ids.extend(generated)
        self.response_mask.extend([1] * len(generated))
        self.response_logprobs.extend(probabilities)

    def append_observation(self, token_ids: list[int], *, limit: int | None = None) -> bool:
        """Append masked transition tokens and return whether all of them fit."""
        observation = list(token_ids)
        maximum = len(observation) if limit is None else max(0, limit)
        appended = observation[:maximum]
        self.token_ids.extend(appended)
        self.response_mask.extend([0] * len(appended))
        self.response_logprobs.extend([0.0] * len(appended))
        return len(appended) == len(observation)
