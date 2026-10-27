from types import SimpleNamespace

import numpy as np

from lsy_drone_racing.control.adaptive_controller import AdaptiveController


def observation() -> dict:
    return dict(
        pos=np.array([-1.0, 0.0, 0.7]),
        vel=np.zeros(3),
        gates_pos=np.array([[0.0, 0.0, 0.7], [1.0, 1.0, 0.7]]),
        gates_quat=np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]),
        gates_visited=np.ones(2, dtype=bool),
        obstacles_pos=np.empty((0, 3)),
        obstacles_visited=np.zeros(0, dtype=bool),
        gate_sequence=np.array([0, 1, 0]),
        gate_sequence_direction=np.array([1, -1, 1]),
        n_gates_passed=0,
    )


def controller(obs: dict) -> AdaptiveController:
    return AdaptiveController(obs, {}, SimpleNamespace(env=SimpleNamespace(freq=50)))


def test_signed_direction_and_repeated_gate():
    obs = observation()
    ctl = controller(obs)
    ctl.index = 1
    gate, _, normal = ctl._geometry(obs)
    assert gate == 1
    np.testing.assert_allclose(normal, [-1.0, 0.0, 0.0])
    ctl.index = 2
    assert ctl._geometry(obs)[0] == 0


def test_unreachable_entry_does_not_start_crossing():
    obs = observation()
    ctl = controller(obs)
    ctl._path = lambda *args: []
    ctl._approach(obs)
    for _ in range(30):
        action = ctl.compute_control(obs)
    assert ctl.blocked
    assert ctl.phase == "approach"
    assert action.shape == (16,)
    assert np.isfinite(action).all()


def test_unknown_gate_does_not_start_crossing():
    obs = observation()
    obs["gates_visited"][:] = False
    ctl = controller(obs)
    ctl._approach(obs)
    ctl.route = []
    obs["pos"] = ctl.goal.copy()
    ctl.compute_control(obs)
    assert ctl.phase == "approach"


def test_crossing_advances_by_sequence_count():
    obs = observation()
    ctl = controller(obs)
    ctl.index = 1
    ctl.phase = "cross"
    ctl.goal = obs["gates_pos"][1] + np.array([-0.24, 0.0, 0.0])
    obs["pos"] = ctl.goal.copy()
    obs["n_gates_passed"] = 2
    ctl.compute_control(obs)
    assert ctl.index == 2
    assert ctl.phase == "approach"
