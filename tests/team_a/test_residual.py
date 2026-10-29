"""Bounded residual tests preserve baseline and verify explicit NumPy schema v2."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from training.team_a.engine import PPOSettings, StagePPOEngine
from training.team_a.stage_export import NumpyStagePolicy
from training.team_contract import RolloutBatch

if TYPE_CHECKING:
    from pathlib import Path


def test_zero_residual_matches_original_and_correction_is_bounded() -> None:
    engine = StagePPOEngine(PPOSettings(policy_kind="residual"))
    x = np.tile(engine.reference.obs_mean.numpy(), (6, 1)).astype(np.float32)
    x[:, 42:] = 0
    stages = np.arange(6, dtype=np.int64)
    np.testing.assert_allclose(
        engine.act(x, stages, True).u, engine.reference.deterministic(x), atol=1e-5
    )
    with torch.no_grad():
        engine.actor.network[-1].bias.fill_(10)
        value = torch.as_tensor(x, dtype=torch.float64)
        correction = engine.actor(value) - engine.reference(value).clamp(-1, 1)
    assert torch.all(correction.abs() <= engine.actor.bounds + 1e-12)


def test_residual_update_keeps_base_frozen_and_roundtrips(tmp_path: Path) -> None:
    engine = StagePPOEngine(PPOSettings(profile="C3-test", policy_kind="residual", epochs=2))
    x = np.tile(engine.reference.obs_mean.numpy(), (16, 1)).astype(np.float32)
    x[:, 42:] = 0
    stages = np.array([0, 1, 2, 3] * 4, dtype=np.int64)
    step = engine.act(x, stages)
    rewards = np.linspace(-1, 1, len(x)).astype(np.float32)
    batch = RolloutBatch(
        x,
        step.z,
        step.u,
        step.log_prob,
        step.value,
        rewards,
        rewards,
        stages,
        step.actor_update_mask,
        "test",
        "C3-test",
    )
    original = [p.detach().clone() for p in engine.actor.base.parameters()]
    metrics = engine.update(batch)
    assert metrics["actor_optimizer_steps"] > 0
    assert all(torch.equal(a, b) for a, b in zip(original, engine.actor.base.parameters()))
    assert all(p.grad is None for p in engine.actor.base.parameters())
    with torch.no_grad():
        engine.actor.network[-1].bias.add_(0.2)
    artifact = tmp_path / "residual.npz"
    manifest = engine.export_npz(
        artifact,
        {
            "run_id": "test",
            "policy_commit": "test",
            "training_commit": "test",
            "training_kind": "synthetic-smoke",
        },
    )
    assert manifest["schema_version"] == "stage-ppo-v2"
    policy = NumpyStagePolicy(artifact, manifest["artifact_sha256"])
    np.testing.assert_allclose(
        policy.commands(x, stages), engine.act(x, stages, True).u, atol=1e-5, rtol=1e-5
    )
    recovered = StagePPOEngine(engine.settings)
    recovered.load_training_state_dict(engine.training_state_dict())
    np.testing.assert_array_equal(recovered.act(x, stages).z, engine.act(x, stages).z)


def test_finetune_checkpoint_from_before_residual_extension_still_loads() -> None:
    engine = StagePPOEngine()
    state = engine.training_state_dict()
    del state["settings"]["policy_kind"]
    del state["settings"]["residual_bounds"]
    recovered = StagePPOEngine()
    recovered.load_training_state_dict(state)
    for original, restored in zip(engine.actor.parameters(), recovered.actor.parameters()):
        assert torch.equal(original, restored)
