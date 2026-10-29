"""Stage routing, latent likelihood, real gradients and checkpoint regression tests."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from training.team_a.engine import PPOSettings, StagePPOEngine
from training.team_a.stage_export import NumpyStagePolicy
from training.team_contract import RolloutBatch

if TYPE_CHECKING:
    from pathlib import Path


def features(engine: StagePPOEngine, count: int) -> np.ndarray:
    """Build reproducible public-feature-shaped inputs, not simulated race rewards."""
    rng = np.random.default_rng(371)
    x = (
        engine.reference.obs_mean.numpy()
        + 0.1 * rng.normal(size=(count, 46)) * engine.reference.obs_std.numpy()
    ).astype(np.float32)
    x[:, 42:46] = 0
    return x


def batch_for(engine: StagePPOEngine, stages: np.ndarray) -> RolloutBatch:
    x = features(engine, len(stages))
    step = engine.act(x, stages)
    # Deliberately synthetic targets exercise the optimizer, not racing quality.
    target = (step.value + np.linspace(-0.4, 1.3, len(stages))).astype(np.float32)
    return RolloutBatch(
        x=x,
        z=step.z,
        u=step.u,
        old_log_prob=step.log_prob,
        old_value=step.value,
        returns=target,
        advantages=(target - step.value).astype(np.float32),
        gate_stage=stages,
        actor_update_mask=step.actor_update_mask,
        run_id="synthetic-test",
        profile=engine.settings.profile,
    )


def parameters(module: torch.nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in module.parameters()]


def test_initial_means_match_and_inactive_stages_are_deterministic() -> None:
    engine = StagePPOEngine()
    x = features(engine, 6)
    stages = np.arange(6, dtype=np.int64)
    baseline = engine.reference.deterministic(x)
    deterministic = engine.act(x, stages, deterministic=True)
    np.testing.assert_allclose(deterministic.u, baseline, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(
        deterministic.actor_update_mask, [True, False, True, False, False, False]
    )
    sample1, sample2 = engine.act(x, stages), engine.act(x, stages)
    inactive = ~sample1.actor_update_mask
    np.testing.assert_array_equal(sample1.z[inactive], sample2.z[inactive])
    np.testing.assert_array_equal(sample1.u[inactive], baseline[inactive])
    np.testing.assert_array_equal(sample1.log_prob[inactive], 0)
    assert np.any(sample1.z[~inactive] != sample2.z[~inactive])
    np.testing.assert_array_equal(sample1.u, np.clip(sample1.z, -1, 1))


def test_likelihood_uses_stored_preclip_latent_even_at_saturation() -> None:
    engine = StagePPOEngine(PPOSettings(initial_std=0.2))
    # Force saturation and make densities at z and clip(z) visibly different.
    with torch.no_grad():
        engine.actor.layers[-1].weight.zero_()
        engine.actor.layers[-1].bias.fill_(2)
    x = features(engine, 8)
    stages = np.zeros(8, dtype=np.int64)
    step = engine.act(x, stages)
    tensor_x, _, _ = engine.tensors(x, stages)
    with torch.no_grad():
        distribution = engine.distribution(tensor_x)
        expected = distribution.log_prob(torch.as_tensor(step.z, dtype=torch.float64)).sum(1)
        wrong = distribution.log_prob(torch.as_tensor(step.u, dtype=torch.float64)).sum(1)
    np.testing.assert_allclose(step.log_prob, expected.numpy(), atol=1e-5, rtol=1e-5)
    assert np.max(np.abs(step.log_prob - wrong.numpy())) > 1


def test_critic_uses_stage_and_act_value_agree() -> None:
    engine = StagePPOEngine()
    x = np.repeat(features(engine, 1), 2, axis=0)
    stages = np.array([0, 2], dtype=np.int64)
    # Route a stage input through the critic explicitly, avoiding probabilistic assertions.
    with torch.no_grad():
        for p in engine.critic.parameters():
            p.zero_()
        engine.critic.network[0].weight[0, 46 + 2] = 1
        engine.critic.network[2].weight[0, 0] = 1
        engine.critic.network[4].weight[0, 0] = 1
    values = engine.value(x, stages)
    assert values[0] == 0 and values[1] > 0
    np.testing.assert_array_equal(engine.act(x, stages).value, values)


def test_update_changes_actor_and_critic_but_keeps_reference_and_inactive_actions() -> None:
    engine = StagePPOEngine(PPOSettings(epochs=2, minibatch_size=8))
    stages = np.array([0, 1, 2, 3] * 4, dtype=np.int64)
    batch = batch_for(engine, stages)
    old_actor, old_critic, old_reference = (
        parameters(engine.actor),
        parameters(engine.critic),
        parameters(engine.reference),
    )
    before = engine.act(batch.x, stages, deterministic=True).u
    metrics = engine.update(batch)
    assert metrics["actor_samples"] == 8
    assert metrics["actor_optimizer_steps"] > 0 and metrics["critic_optimizer_steps"] == 4
    assert any(not torch.equal(a, b) for a, b in zip(old_actor, engine.actor.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(old_critic, engine.critic.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(old_reference, engine.reference.parameters()))
    assert all(p.grad is None and not p.requires_grad for p in engine.reference.parameters())
    after = engine.act(batch.x, stages, deterministic=True).u
    np.testing.assert_array_equal(after[~batch.actor_update_mask], before[~batch.actor_update_mask])
    assert np.isfinite([v for v in metrics.values() if isinstance(v, (float, int))]).all()


def test_all_masked_batch_only_updates_critic_and_not_exploration() -> None:
    engine = StagePPOEngine(PPOSettings(epochs=1, minibatch_size=4))
    batch = batch_for(engine, np.array([1, 3, 4, 5], dtype=np.int64))
    before, old_std = parameters(engine.actor), engine.log_std.detach().clone()
    metrics = engine.update(batch)
    assert metrics["actor_samples"] == 0 and metrics["actor_optimizer_steps"] == 0
    assert metrics["critic_optimizer_steps"] == 1
    assert all(torch.equal(a, b) for a, b in zip(before, engine.actor.parameters()))
    assert torch.equal(old_std, engine.log_std)
    assert not engine.actor_optimizer.state


def test_reference_anchor_has_corrective_gradient() -> None:
    engine = StagePPOEngine()
    x = torch.as_tensor(features(engine, 8), dtype=torch.float64)
    with torch.no_grad():
        engine.actor.layers[-1].weight.zero_()
        engine.actor.layers[-1].bias.zero_()
        reference = engine.reference(x).clamp(-1, 1)
    loss = (engine.actor(x).clamp(-1, 1) - reference).square().mean()
    loss.backward()
    gradient = engine.actor.layers[-1].bias.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert torch.any(gradient != 0)


def test_checkpoint_restores_optimizer_rng_and_next_update_exactly(tmp_path: Path) -> None:
    settings = PPOSettings(epochs=1, minibatch_size=8)
    engine = StagePPOEngine(settings)
    engine.update(batch_for(engine, np.array([0, 1, 2, 3] * 4, dtype=np.int64)))
    state = engine.training_state_dict()
    checkpoint = tmp_path / "trusted-local-checkpoint.pt"
    torch.save(state, checkpoint)
    restored = StagePPOEngine(settings)
    restored.load_training_state_dict(torch.load(checkpoint, weights_only=True))
    stages = np.array([0, 1, 2, 3] * 4, dtype=np.int64)
    x = features(engine, len(stages))
    step1, step2 = engine.act(x, stages), restored.act(x, stages)
    for field in ["mu", "z", "u", "log_prob", "value", "actor_update_mask"]:
        np.testing.assert_array_equal(getattr(step1, field), getattr(step2, field))
    batch = batch_for(engine, stages)
    restored_batch = batch_for(restored, stages)
    np.testing.assert_array_equal(batch.z, restored_batch.z)
    result1, result2 = engine.update(batch), restored.update(restored_batch)
    assert result1["update_count"] == result2["update_count"]
    for a, b in zip(engine.actor.parameters(), restored.actor.parameters()):
        assert torch.equal(a, b)
    for a, b in zip(engine.critic.parameters(), restored.critic.parameters()):
        assert torch.equal(a, b)
    frozen_copy = state["actor"]["layers.0.weight"].clone()
    with torch.no_grad():
        engine.actor.layers[0].weight.add_(1)
    assert torch.equal(state["actor"]["layers.0.weight"], frozen_copy)


def test_numpy_stage_export_after_update_and_changed_actor(tmp_path: Path) -> None:
    engine = StagePPOEngine(PPOSettings(epochs=1))
    engine.update(batch_for(engine, np.array([0, 1, 2, 3] * 4, dtype=np.int64)))
    with torch.no_grad():
        engine.actor.layers[-1].weight.zero_()
        engine.actor.layers[-1].bias.copy_(torch.tensor([0.2, -0.3, 0.4, -0.5]))
    artifact = tmp_path / "stage.npz"
    manifest = engine.export_npz(
        artifact,
        {
            "run_id": "synthetic-test",
            "policy_commit": "test",
            "training_commit": "test",
            "training_kind": "synthetic-smoke",
        },
    )
    policy = NumpyStagePolicy(artifact, manifest["artifact_sha256"])
    x, stages = features(engine, 6), np.arange(6, dtype=np.int64)
    np.testing.assert_allclose(
        policy.commands(x, stages),
        engine.act(x, stages, deterministic=True).u,
        rtol=1e-5,
        atol=1e-5,
    )
    assert manifest["active_stages"] == [0, 2]
    assert manifest["training_manifest"]["training_kind"] == "synthetic-smoke"
    with pytest.raises(FileExistsError):
        engine.export_npz(artifact, manifest["training_manifest"])


@pytest.mark.parametrize("error", ["mask", "profile", "contract", "u", "nan", "stage"])
def test_invalid_rollout_rejected_before_any_update(error: str) -> None:
    engine = StagePPOEngine()
    batch = batch_for(engine, np.array([0, 1, 2, 3], dtype=np.int64))
    if error == "mask":
        batch = replace(batch, actor_update_mask=~batch.actor_update_mask)
    elif error == "profile":
        batch = replace(batch, profile="different-transition-rule")
    elif error == "contract":
        batch = replace(batch, contract_version="v1")
    elif error == "u":
        batch = replace(batch, u=np.full_like(batch.u, 9))
    elif error == "nan":
        batch = replace(batch, returns=np.full_like(batch.returns, np.nan))
    else:
        batch = replace(batch, gate_stage=np.full_like(batch.gate_stage, 100))
    old_actor, old_critic = parameters(engine.actor), parameters(engine.critic)
    with pytest.raises(ValueError):
        engine.update(batch)
    assert engine.update_count == 0
    assert all(torch.equal(a, b) for a, b in zip(old_actor, engine.actor.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(old_critic, engine.critic.parameters()))


def test_excessive_kl_stops_actor_but_fits_critic() -> None:
    engine = StagePPOEngine(PPOSettings(epochs=1, minibatch_size=8))
    batch = batch_for(engine, np.zeros(8, dtype=np.int64))
    batch = replace(batch, old_log_prob=batch.old_log_prob + np.float32(5))
    old_actor = parameters(engine.actor)
    metrics = engine.update(batch)
    assert metrics["early_stopped_kl"] and metrics["actor_optimizer_steps"] == 0
    assert metrics["critic_optimizer_steps"] == 1
    assert all(torch.equal(a, b) for a, b in zip(old_actor, engine.actor.parameters()))


def test_constructor_does_not_consume_global_torch_rng() -> None:
    torch.manual_seed(17)
    state = torch.random.get_rng_state().clone()
    StagePPOEngine()
    assert torch.equal(state, torch.random.get_rng_state())
