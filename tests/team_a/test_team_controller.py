"""Controller load, NumPy-only deployment, reset and baseline behavior checks."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.level2_reference import Level2Reference
from lsy_drone_racing.control.team_policy_controller import TeamPolicyController
from lsy_drone_racing.utils import load_config, load_controller
from training.team_a.engine import PPOSettings, StagePPOEngine
from training.team_a.policy import encode_obs, from_environment_action, to_environment_action
from training.team_a.stage_export import export_stage_actor

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def observation() -> dict:
    config = load_config(ROOT / "config/level2.toml")
    gates = config.env.track.gates
    return {
        "pos": np.array([-1.5, 0.75, 0.01]),
        "quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "vel": np.array([0.2, -0.3, 0.1]),
        "ang_vel": np.zeros(3),
        "gates_pos": np.array([gate["pos"] for gate in gates]),
        "gates_quat": Rotation.from_euler(
            "xyz", np.array([gate["rpy"] for gate in gates])
        ).as_quat(),
        "gate_sequence": np.array([0, 1, 2, 3, 1]),
        "gate_sequence_direction": np.array([1, 1, 1, 1, -1]),
        "n_gates_passed": 0,
        "gates_visited": np.array([False, True, False, False]),
        "obstacles_pos": np.array([pole["pos"] for pole in config.env.track.obstacles]),
        "obstacles_visited": np.array([False, True, False, False]),
    }


@pytest.mark.parametrize(
    "filename, classname",
    [
        ("team_policy_controller.py", "TeamPolicyController"),
        ("team_c3_pilot_controller.py", "PilotController"),
    ],
)
def test_official_loader_discovers_exactly_one_controller(filename: str, classname: str) -> None:
    selected = load_controller(ROOT / "lsy_drone_racing/control" / filename)
    assert selected.__name__ == classname


def test_ten_step_entry_matches_baseline_and_training_history(observation: dict) -> None:
    config = load_config(ROOT / "config/level2.toml")
    candidate = TeamPolicyController(observation, {}, config)
    baseline = Level2Reference(observation, {}, config)
    engine = StagePPOEngine()
    previous = np.zeros((1, 4), dtype=np.float32)
    for step in range(10):
        observation["n_gates_passed"] = step // 2
        observation["pos"] += np.array([0.02, -0.01, 0.005])
        x = encode_obs([observation], previous)
        u = engine.act(x, np.array([step // 2], dtype=np.int64), deterministic=True).u
        training_action = to_environment_action(u, candidate.thrust_low, candidate.thrust_high)
        action = candidate.compute_control(observation)
        np.testing.assert_allclose(
            action, baseline.compute_control(observation), atol=1e-5, rtol=1e-5
        )
        np.testing.assert_allclose(action, training_action[0], atol=1e-5, rtol=1e-5)
        assert np.all(action >= candidate.space.low) and np.all(action <= candidate.space.high)
        assert candidate.step_callback(action, observation, 0.0, False, False, {}) is False
        previous = from_environment_action(
            training_action, candidate.thrust_low, candidate.thrust_high
        )
        np.testing.assert_allclose(candidate.previous, previous[0], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("hook", ["reset", "episode_reset", "episode_callback"])
def test_reset_hooks_do_not_touch_other_instance(observation: dict, hook: str) -> None:
    config = load_config(ROOT / "config/level2.toml")
    a = TeamPolicyController(observation, {}, config)
    b = TeamPolicyController(observation, {}, config)
    a.compute_control(observation)
    b.compute_control(observation)
    remembered = b.previous.copy()
    getattr(a, hook)()
    np.testing.assert_array_equal(a.previous, 0)
    np.testing.assert_array_equal(b.previous, remembered)
    fresh = TeamPolicyController(observation, {}, config)
    np.testing.assert_array_equal(
        a.compute_control(observation), fresh.compute_control(observation)
    )


@pytest.mark.parametrize("case", ["interface", "freq", "level3", "sequence"])
def test_unsupported_conditions_are_rejected(observation: dict, case: str) -> None:
    config = load_config(ROOT / "config/level2.toml")
    if case == "interface":
        config.env.control_mode = "state"
    elif case == "freq":
        config.env.freq = 100
    elif case == "level3":
        config.env.track.randomize = True
    else:
        observation["gate_sequence"] = np.array([0, 1])
    with pytest.raises(ValueError):
        TeamPolicyController(observation, {}, config)


def test_bad_pinned_hash_is_rejected(observation: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TeamPolicyController, "POLICY_SHA256", "incorrect")
    config = load_config(ROOT / "config/level2.toml")
    with pytest.raises(ValueError, match="SHA256"):
        TeamPolicyController(observation, {}, config)


def test_synthetic_weights_rejected_even_with_matching_manifest(
    observation: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Point only the test class's assets at a synthetic package inside policies_team.
    # No asset is copied into the real source tree; replace the runtime loader instead.
    from lsy_drone_racing.control import team_policy_controller as module
    from lsy_drone_racing.control.team_policy_runtime import NumpyStagePolicy

    engine = StagePPOEngine(PPOSettings())
    artifact = tmp_path / "synthetic.npz"
    manifest = export_stage_actor(
        engine,
        artifact,
        {
            "run_id": "test",
            "policy_commit": "test",
            "training_commit": "test",
            "training_kind": "synthetic-smoke",
        },
    )
    actual_policy = NumpyStagePolicy(artifact, manifest["artifact_sha256"])
    original_read_text = Path.read_text

    def read_manifest(path: Path, *args: object, **kwargs: object) -> str:
        if path.name == TeamPolicyController.MANIFEST_FILE:
            return json.dumps(manifest)
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(module, "NumpyStagePolicy", lambda *_: actual_policy)
    monkeypatch.setattr(TeamPolicyController, "POLICY_SHA256", manifest["artifact_sha256"])
    monkeypatch.setattr(Path, "read_text", read_manifest)
    config = load_config(ROOT / "config/level2.toml")
    with pytest.raises(ValueError, match="Synthetic"):
        TeamPolicyController(observation, {}, config)


def test_deployment_imports_no_torch_or_training_package() -> None:
    code = (
        "import sys; "
        "from lsy_drone_racing.control.team_policy_controller import TeamPolicyController; "
        "assert 'torch' not in sys.modules; "
        "assert not any(n == 'training' or n.startswith('training.') for n in sys.modules); "
        "print('NUMPY_DEPLOYMENT_IMPORT_OK')"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    environment["SCIPY_ARRAY_API"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", code], env=environment, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert "NUMPY_DEPLOYMENT_IMPORT_OK" in result.stdout


def test_modified_actor_cannot_be_labeled_baseline(tmp_path: Path) -> None:
    import torch

    engine = StagePPOEngine()
    with torch.no_grad():
        engine.actor.layers[-1].bias.add_(0.1)
    with pytest.raises(ValueError, match="unchanged"):
        export_stage_actor(
            engine,
            tmp_path / "invalid.npz",
            {
                "run_id": "invalid",
                "policy_commit": "test",
                "training_commit": "test",
                "training_kind": "baseline-conversion",
            },
        )
