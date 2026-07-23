"""Small, dependency-free rollout policy predicates."""

from __future__ import annotations

import re


def looks_like_focused_test(action: str) -> bool:
    """Return whether a successful command is meaningful enough to finalize."""
    lowered = action.lower()
    if "|" in action or action.lstrip().startswith(("STR_",)):
        return False
    markers = (
        "pytest",
        "python -m unittest",
        "python -m django test",
        "manage.py test",
        "runtests.py",
        "tox ",
        "tox -",
    )
    scratch_reproducer = re.search(
        r"\bpython(?:\d+(?:\.\d+)*)?\s+(?:\S*/)?(?:test|repro|check)[^/\s]*\.py(?:\s|$)",
        lowered,
    )
    return any(marker in lowered for marker in markers) or scratch_reproducer is not None


def looks_like_environment_mutation(action: str) -> bool:
    """Return whether a command attempts to mutate the fixed task environment."""
    lowered = " ".join(action.lower().split())
    markers = (
        "apt-get install",
        "apt install",
        "yum install",
        "dnf install",
        "pip install",
        "python -m pip install",
        "python3 -m pip install",
        "conda install",
        "mamba install",
    )
    return any(marker in lowered for marker in markers)
