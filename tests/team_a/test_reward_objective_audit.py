"""Guard the reward-audit claims, without modifying or training a race policy."""

from copy import deepcopy

import numpy as np

from training.team_b.reward import racing_reward
from validation.team_a.reward_objective_audit import (
    potential,
    potential_terms,
    progress_decomposition,
    variant_scores,
)


def observation(x: float, gate_x: float = 1.0, stage: int = 0) -> dict:
    """One-dimensional scripted accounting state, not a simulated race result."""
    return {
        "pos": [x, 0, 0],
        "gates_pos": [[gate_x, 0, 0]],
        "gate_sequence": [0],
        "n_gates_passed": stage,
    }


def script() -> list[dict]:
    """A revelation, crossing and terminal warp test the telescoping boundaries."""
    states = [
        observation(0),
        observation(0.4, 1.2),
        observation(0.8, 1.2),
        observation(1.25, 1.2, 1),
        observation(-1, 1.2, 1),
    ]
    return [
        {
            "obs": states[i],
            "next_obs": states[i + 1],
            "gate_stage": min(i // 3, 1),
            "terminated": i == 3,
            "truncated": False,
        }
        for i in range(4)
    ]


def test_revelation_changes_telescoping_boundary_not_instantaneous_motion_reward() -> None:
    trace = script()
    progress = sum(
        racing_reward(
            step["obs"], step["next_obs"], np.zeros(4), np.zeros(4), terminated=step["terminated"]
        )[1]["progress"]
        for step in trace
    )
    result = progress_decomposition(trace, progress)
    np.testing.assert_allclose(progress, 1.15)
    np.testing.assert_allclose(result["same_target_observation_update_correction"], 0.2)


def test_discount_consistent_potential_is_initial_state_offset_even_with_warp() -> None:
    trace = script()
    weights = 0.99 ** np.arange(len(trace))
    np.testing.assert_allclose(weights @ potential_terms(trace), -potential(trace[0]["obs"]))
    failure = deepcopy(trace[:1])
    failure[0].update(terminated=True)
    failure[0]["next_obs"]["pos"] = [-1, -1, -1]
    np.testing.assert_allclose(potential_terms(failure), [-potential(failure[0]["obs"])])


def test_fixed_target_closed_loop_does_not_earn_net_progress() -> None:
    states = [observation(x) for x in (0, 0.4, 0.2, 0)]
    values = [
        racing_reward(a, b, np.zeros(4), np.zeros(4))[1]["progress"]
        for a, b in zip(states[:-1], states[1:])
    ]
    np.testing.assert_allclose(sum(values), 0, atol=1e-12)


def test_large_time_weight_can_rank_early_failure_above_success() -> None:
    successful = np.zeros((170, 6))
    successful[:, 3] = -0.02
    successful[[30, 60, 90, 120, 168], 1] = 5
    successful[168, 2] = 20
    failed = np.zeros((35, 6))
    failed[:, 3] = -0.02
    failed[-1, 4] = -5
    good = variant_scores(successful, np.zeros(170))
    bad = variant_scores(failed, np.zeros(35))
    assert good["current"] > bad["current"]
    assert good["time_x100"] < bad["time_x100"]
