"""Check observational diagnostics without opening a new environment scene."""

from copy import deepcopy

import numpy as np

from validation.team_a.full_diagnosis import crossing, matched, reward_order_audit


def plane_step() -> dict:
    """Scripted first-gate crossing with an unrevealed next gate."""
    obs = {
        "pos": [-0.1, 0, 0],
        "vel": [1, 0, 0],
        "gate_sequence": [0, 1],
        "gate_sequence_direction": [1, 1],
        "gates_pos": [[0, 0, 0], [1, 0, 0]],
        "gates_quat": [[0, 0, 0, 1], [0, 0, 0, 1]],
        "gates_visited": [True, False],
        "n_gates_passed": 0,
    }
    after = deepcopy(obs)
    after.update(pos=[0.1, 0, 0], n_gates_passed=1)
    return {
        "step": 3,
        "gate_stage": 0,
        "obs": obs,
        "next_obs": after,
        "terminated": False,
        "truncated": False,
    }


def test_interpolation_keeps_next_geometry_knowledge_flag() -> None:
    """Nominal next-gate direction is never relabeled as revealed geometry."""
    result = crossing(plane_step())
    np.testing.assert_allclose(result["position"], [0, 0, 0])
    np.testing.assert_allclose(result["event_time_s"], 0.07)
    assert not result["next_gate_known"]
    assert result["normal_speed_mps"] == 1


def test_terminal_warp_is_not_interpolated() -> None:
    step = plane_step()
    step["terminated"] = True
    step["next_obs"]["pos"] = [-1, -1, -1]
    assert crossing(step) is None


def test_geometry_does_not_override_official_gate_counter() -> None:
    step = plane_step()
    step["next_obs"]["n_gates_passed"] = 0
    assert crossing(step) is None


def test_reward_ranking_uses_same_seed_and_complete_successes() -> None:
    data = [
        {
            "seed": 42,
            "success": True,
            "update": 1,
            "official_time": 3.0,
            "training_reward": 50.0,
            "reward_components": {"progress": 8.0},
        },
        {
            "seed": 42,
            "success": True,
            "update": 2,
            "official_time": 3.04,
            "training_reward": 50.1,
            "reward_components": {"progress": 8.14},
        },
        {
            "seed": 43,
            "success": True,
            "update": 3,
            "official_time": 4.0,
            "training_reward": 100.0,
            "reward_components": {},
        },
        {
            "seed": 42,
            "success": False,
            "update": 4,
            "official_time": 1.0,
            "training_reward": 0.0,
            "reward_components": {},
        },
    ]
    result = reward_order_audit(data)
    assert result["same_seed_successful_pairs_at_least_40ms_apart"] == 1
    assert result["slower_higher_reward_pairs"] == 1


def test_discordant_success_test_and_no_common_timing() -> None:
    base = {i: {"success": i >= 7} for i in range(10)}
    other = {i: {"success": i < 7} for i in range(10)}
    result = matched(base, other)
    assert result["paired_success_exact_two_sided_p"] == 0.34375
    assert result["time_delta_s"] == {"n": 0}
    assert len(result["recovered"]) == 7 and len(result["lost"]) == 3
