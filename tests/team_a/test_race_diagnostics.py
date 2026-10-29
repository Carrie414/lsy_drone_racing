"""Protect diagnostic time accounting from local-gain and final-step errors."""

import numpy as np

from validation.team_a.diagnose_race import durations


def episode(first_boundary: int = 40) -> dict:
    """A scripted accounting example, never a simulated performance result."""
    boundaries = [first_boundary, 76, 108, 144, 174]
    return {
        "control_steps": 175,
        "transition_commands": [
            {"step": step, "to_stage": stage + 1} for stage, step in enumerate(boundaries)
        ],
    }


def test_stage_intervals_include_post_completion_step() -> None:
    """Elapsed intervals and official final-step time differ by exactly one step."""
    result = durations(episode())
    np.testing.assert_allclose(result, [0.8, 0.72, 0.64, 0.72, 0.6, 0.02])
    np.testing.assert_allclose(result.sum(), 175 / 50)
    np.testing.assert_allclose(result.sum() - 174 / 50, 0.02)


def test_earlier_first_gate_does_not_imply_faster_total() -> None:
    """Moving one gate boundary earlier can transfer time to its following stage."""
    delta = durations(episode(39)) - durations(episode())
    np.testing.assert_allclose(delta[:2], [-0.02, 0.02])
    np.testing.assert_allclose(delta.sum(), 0, atol=1e-12)
