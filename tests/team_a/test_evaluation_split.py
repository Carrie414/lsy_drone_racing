"""Check split boundaries without opening a reserved scene or simulator."""

import pytest

from validation.team_b.paired_eval import validate_split


@pytest.mark.parametrize(
    "seed,runs,split,allowed",
    [
        (110001, 200, "development", True),
        (120001, 100, "validation", True),
        (120000, 2, "development", False),
        (120001, 101, "validation", False),
        (130001, 1, "development", False),
        (110001, 100, "validation", False),
    ],
)
def test_reserved_boundaries(seed: int, runs: int, split: str, allowed: bool) -> None:
    """Validation needs explicit intent and cannot spill into other seed ranges."""
    if allowed:
        validate_split(seed, runs, split)
    else:
        with pytest.raises(ValueError):
            validate_split(seed, runs, split)
