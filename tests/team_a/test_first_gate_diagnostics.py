"""Guard first-gate actionable-revelation timing and nominal-geometry labels."""

from copy import deepcopy

import pytest

from validation.team_b.first_gate_diagnostics import geometry, summarize_episode


def observation(visited: bool = False, passed: int = 0) -> dict:
    """A public state approaching the identity gate plane."""
    return {
        "pos": [-0.4, 0.1, 0.2],
        "vel": [2.0, -0.25, -0.5],
        "gates_pos": [[0.0, 0.0, 0.0]],
        "gates_quat": [[0.0, 0.0, 0.0, 1.0]],
        "gates_visited": [visited],
        "gate_sequence": [0],
        "n_gates_passed": passed,
    }


def test_projection_is_signed_in_gate_frame():
    """Project only future plane intersections; receding motion has no prediction."""
    obs = observation()
    result = geometry(obs, 0)
    assert result["plane_eta_s"] == pytest.approx(0.2)
    assert result["projected_lateral_m"] == pytest.approx(0.05)
    assert result["projected_vertical_m"] == pytest.approx(0.1)
    obs["vel"][0] = -2.0
    assert geometry(obs, 0)["plane_eta_s"] is None


def test_terminal_revelation_has_no_available_correction_step():
    """A gate appearing only after the final command cannot guide that command."""
    before, after = observation(), observation(True)
    trace = [{"step": 0, "gate_stage": 0, "obs": before, "next_obs": after, "action": [0, 0, 0, 0]}]
    episode = {"eval_seed": 1, "success": False}
    result = summarize_episode(episode, trace)
    assert result["terminal_only_reveal"]
    assert result["correction_steps"] == 0
    assert not result["last_geometry_revealed"]
    second = deepcopy(trace[0])
    second.update(step=1, obs=after, next_obs=observation(True, 1))
    result = summarize_episode(episode, trace + [second])
    assert result["first_gate_passed"]
    assert result["correction_steps"] == 1
    assert not result["terminal_only_reveal"]
