"""A0 behavior-preservation tests against the real pretrained baseline."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.level2_reference import Level2Reference, reference_features
from lsy_drone_racing.utils import load_config
from training.team_a.export_npz import export_actor
from training.team_a.npz_io import ORIGINAL_MODEL, ORIGINAL_SHA256, load_actor_arrays, numpy_mean
from training.team_a.policy import (
    PretrainedActor,
    encode_obs,
    from_environment_action,
    to_environment_action,
)
from training.team_contract import CONTRACT_VERSION, PolicyStep, RolloutBatch

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def observation() -> dict:
    return {
        "pos": np.array([-1.5, 0.75, 0.01]),
        "quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "vel": np.array([0.2, -0.3, 0.1]),
        "ang_vel": np.zeros(3),
        "gates_pos": np.array([[0.5, 0.25, 0.7], [1.05, 0.75, 1.2]]),
        "gates_quat": Rotation.from_euler("z", np.array([[-0.78], [2.35]])).as_quat(),
        "gate_sequence": np.array([0, 1, 0]),
        "gate_sequence_direction": np.array([1, 1, -1]),
        "n_gates_passed": 0,
        "gates_visited": np.array([False, True]),
        "obstacles_pos": np.array([[0.0, 0.75, 1.55], [1.0, 0.25, 1.55]]),
        "obstacles_visited": np.array([False, True]),
    }


@pytest.fixture(scope="module")
def actor() -> PretrainedActor:
    return PretrainedActor()


def test_original_checksum_and_architecture(actor: PretrainedActor) -> None:
    arrays = load_actor_arrays(ORIGINAL_MODEL, ORIGINAL_SHA256)
    assert actor.tilt_max == 1.5
    for index, layer in enumerate(actor.layers):
        np.testing.assert_array_equal(layer.weight.detach().numpy(), arrays[f"W{index}"].T)
        np.testing.assert_array_equal(layer.bias.detach().numpy(), arrays[f"b{index}"])


@pytest.mark.parametrize("stage", [0, 1, 2])
@pytest.mark.parametrize("empty_poles", [False, True])
def test_encoder_matches_signed_repeated_gate_fixture(
    observation: dict, stage: int, empty_poles: bool
) -> None:
    observation["n_gates_passed"] = stage
    if empty_poles:
        observation["obstacles_pos"] = np.empty((0, 3))
        observation["obstacles_visited"] = np.empty(0, dtype=bool)
    before = copy.deepcopy(observation)
    previous = np.array([[0.1, 0.2, -0.1, -0.3]], dtype=np.float32)
    encoded = encode_obs([observation], previous)
    np.testing.assert_array_equal(encoded[0], reference_features(observation, previous[0]))
    for key, value in before.items():
        np.testing.assert_array_equal(observation[key], value)


def test_raw_means_and_export_agree_on_normalization_boundaries(
    actor: PretrainedActor, tmp_path: Path
) -> None:
    arrays = load_actor_arrays(ORIGINAL_MODEL, ORIGINAL_SHA256)
    rng = np.random.default_rng(7081)
    normalized = np.vstack(
        [rng.normal(size=(64, 46)), np.full((1, 46), -12.0), np.full((1, 46), 12.0)]
    )
    features = (arrays["obs_mean"] + normalized * arrays["obs_std"]).astype(np.float32)
    baseline = numpy_mean(features, arrays)
    with torch.no_grad():
        converted = actor(torch.from_numpy(features)).numpy()
    np.testing.assert_allclose(converted, baseline, rtol=1e-5, atol=1e-5)
    output = tmp_path / "roundtrip.npz"
    manifest = export_actor(actor, output)
    assert manifest["stage"] == "A0-conversion-only-not-trained"
    exported = load_actor_arrays(output, manifest["artifact_sha256"])
    np.testing.assert_allclose(numpy_mean(features, exported), baseline, rtol=1e-5, atol=1e-5)
    reimported = PretrainedActor(output, expected_sha256=manifest["artifact_sha256"])
    np.testing.assert_allclose(
        reimported.deterministic(features), actor.deterministic(features), rtol=1e-5, atol=1e-5
    )


def test_ten_step_actions_and_final_command_history(
    actor: PretrainedActor, observation: dict, tmp_path: Path
) -> None:
    config = load_config(ROOT / "config/level2.toml")
    baseline = Level2Reference(observation, {}, config)
    arrays = load_actor_arrays(ORIGINAL_MODEL, ORIGINAL_SHA256)
    export = tmp_path / "history.npz"
    export_actor(actor, export)
    exported = load_actor_arrays(export)
    previous = np.zeros((1, 4), dtype=np.float32)
    numpy_previous = np.zeros_like(previous)
    for step in range(10):
        observation["n_gates_passed"] = min(step // 4, 2)
        observation["pos"] = observation["pos"] + np.array([0.03, -0.01, 0.005])
        expected = baseline.compute_control(observation)
        features = encode_obs([observation], previous)
        command = actor.deterministic(features)
        action = to_environment_action(command, baseline.thrust_low, baseline.thrust_high)
        np.testing.assert_allclose(action[0], expected, rtol=1e-5, atol=1e-5)
        numpy_command = np.clip(
            numpy_mean(encode_obs([observation], numpy_previous), exported), -1, 1
        )
        numpy_action = to_environment_action(
            numpy_command, baseline.thrust_low, baseline.thrust_high
        )
        np.testing.assert_allclose(numpy_action, action, rtol=1e-5, atol=1e-5)
        previous = from_environment_action(action, baseline.thrust_low, baseline.thrust_high)
        numpy_previous = from_environment_action(
            numpy_action, baseline.thrust_low, baseline.thrust_high
        )
        # Check inverse mapping error independently of the recurrent forward comparison.
        np.testing.assert_allclose(previous, command, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(arrays["W0"], load_actor_arrays(ORIGINAL_MODEL)["W0"])


def test_history_is_explicit_and_one_environment_can_reset(observation: dict) -> None:
    previous = np.array([[0.1, 0.2, 0.3, 0.4], [-0.1, -0.2, -0.3, -0.4]], dtype=np.float32)
    before = encode_obs([observation, observation], previous)
    previous[0] = 0  # The caller resets only the finished instance.
    after = encode_obs([observation, observation], previous)
    np.testing.assert_array_equal(after[0, 42:], 0)
    np.testing.assert_array_equal(after[1], before[1])


def test_action_mapping_bounds_and_clipping() -> None:
    commands = np.array([[-2, 2, 0, -3], [1, -1, 1, 3]], dtype=np.float32)
    actions = to_environment_action(commands, 0.1, 0.8)
    np.testing.assert_allclose(actions, [[-1.5, 1.5, 0, 0.1], [1.5, -1.5, 0.5, 0.8]])
    np.testing.assert_allclose(
        from_environment_action(actions, 0.1, 0.8), np.clip(commands, -1, 1), atol=1e-6
    )


def test_invalid_inputs_and_baseline_overwrite_are_rejected(
    actor: PretrainedActor, observation: dict, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="SHA256"):
        load_actor_arrays(ORIGINAL_MODEL, "not-the-model-hash")
    with pytest.raises(ValueError, match="original baseline"):
        export_actor(actor, ORIGINAL_MODEL)
    output = tmp_path / "immutable.npz"
    export_actor(actor, output)
    with pytest.raises(FileExistsError):
        export_actor(actor, output)
    for invalid in [np.zeros((2, 45)), np.full((1, 46), np.nan)]:
        with pytest.raises(ValueError, match="Features"):
            actor(torch.as_tensor(invalid))
    with pytest.raises(ValueError, match="previous_u"):
        encode_obs([observation], np.zeros((2, 4)))
    arrays = load_actor_arrays(ORIGINAL_MODEL)
    arrays["obs_std"][0] = 0
    malformed = tmp_path / "zero_scale.npz"
    np.savez(malformed, **arrays)
    with pytest.raises(ValueError, match="positive"):
        load_actor_arrays(malformed)


def test_actor_retains_finite_training_gradients(actor: PretrainedActor) -> None:
    actor.zero_grad(set_to_none=True)
    x = torch.tensor(actor.obs_mean.detach().numpy()[None, :], dtype=torch.float64)
    actor(x).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters())
    actor.zero_grad(set_to_none=True)


def test_contract_records_have_v2_fields() -> None:
    assert CONTRACT_VERSION == "v2"
    assert {"mu", "z", "u", "value", "actor_update_mask"} <= PolicyStep.__dataclass_fields__.keys()
    assert {"gate_stage", "returns", "run_id", "contract_version"} <= (
        RolloutBatch.__dataclass_fields__.keys()
    )
