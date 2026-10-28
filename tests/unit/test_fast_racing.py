"""Regression checks for signed observations and the frozen public actor adapter."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.fast_racing_controller import FastRacingController
from lsy_drone_racing.control.level2_reference import Level2Reference, reference_features
from lsy_drone_racing.utils import load_config

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def observation() -> dict:
    """Include a repeated gate and reverse pass in one compact public observation."""
    return dict(
        pos=np.array([-1.5, 0.75, 0.01]),
        quat=np.array([0.0, 0.0, 0.0, 1.0]),
        vel=np.array([0.2, -0.3, 0.1]),
        ang_vel=np.zeros(3),
        gates_pos=np.array([[0.5, 0.25, 0.7], [1.05, 0.75, 1.2]]),
        gates_quat=Rotation.from_euler("z", np.array([[-0.78], [2.35]])).as_quat(),
        gate_sequence=np.array([0, 1, 0]),
        gate_sequence_direction=np.array([1, 1, -1]),
        n_gates_passed=0,
        gates_visited=np.array([False, True]),
        obstacles_pos=np.array([[0.0, 0.75, 1.55], [1.0, 0.25, 1.55]]),
        obstacles_visited=np.array([False, True]),
    )


def test_public_feature_layout_and_input_immutability(observation: dict):
    """Features preserve the observed gate centers, sensor flags and action history."""
    before = {k: np.asarray(v).copy() for k, v in observation.items()}
    previous = np.array([0.1, 0.2, -0.1, -0.3])
    encoded = reference_features(observation, previous)
    assert encoded.shape == (46,)
    np.testing.assert_allclose(encoded[15:18], observation["gates_pos"][0] - observation["pos"])
    np.testing.assert_allclose(encoded[-4:], previous)
    assert encoded[24] == 0
    for key in before:
        np.testing.assert_array_equal(before[key], observation[key])


def test_repeated_reverse_gate_and_last_gate_mask(observation: dict):
    """Sequence slots, rather than physical gate IDs, control the target and sign."""
    observation["n_gates_passed"] = 2
    encoded = reference_features(observation, np.zeros(4))
    normal = Rotation.from_quat(observation["gates_quat"][0]).as_matrix()[:, 0]
    np.testing.assert_allclose(encoded[18:21], -normal, atol=1e-6)
    np.testing.assert_array_equal(encoded[25:36], 0)


def test_no_poles_padding(observation: dict):
    """Missing poles do not change the trained network's input dimension."""
    observation["obstacles_pos"] = np.empty((0, 3))
    observation["obstacles_visited"] = np.empty(0, dtype=bool)
    encoded = reference_features(observation, np.zeros(4))
    assert encoded.shape == (46,)
    np.testing.assert_array_equal(encoded[36:42], 0)


def test_validated_entry_is_numerically_identical_to_reference(observation: dict):
    """The checksum/usage guard must not change any action from the evaluated adapter."""
    config = load_config(ROOT / "config/level2.toml")
    controller = FastRacingController(observation, {}, config)
    reference = Level2Reference(observation, {}, config)
    for _ in range(8):
        action = controller.compute_control(observation)
        np.testing.assert_array_equal(action, reference.compute_control(observation))
        assert action.shape == (4,) and np.isfinite(action).all()
        assert np.all(np.abs(action[:2]) <= controller.tilt_max + 1e-6)
        assert abs(action[2]) <= 0.5
        assert controller.thrust_low <= action[3] <= controller.thrust_high
    controller.episode_callback()
    np.testing.assert_array_equal(controller.previous, 0)


def test_random_track_use_is_rejected(observation: dict):
    """Do not silently present a fixed-layout racing policy as a Level 3 solution."""
    config = load_config(ROOT / "config/level3.toml")
    with pytest.raises(ValueError, match="Level 3"):
        FastRacingController(observation, {}, config)


def test_wrong_command_interface_is_rejected(observation: dict):
    """Four-component attitude actions must never be sent to the state interface."""
    config = load_config(ROOT / "config/level2.toml")
    config.env.control_mode = "state"
    with pytest.raises(ValueError, match="attitude"):
        FastRacingController(observation, {}, config)
