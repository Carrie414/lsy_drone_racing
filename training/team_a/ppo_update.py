"""PPO updates with actor masks, frozen-reference anchoring and stage-aware values."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from training.team_contract import CONTRACT_VERSION

if TYPE_CHECKING:
    from training.team_a.engine import StagePPOEngine
    from training.team_contract import RolloutBatch


def update_engine(engine: StagePPOEngine, batch: RolloutBatch) -> dict:
    """Apply PPO to active stages and value regression to the complete rollout."""
    if batch.contract_version != CONTRACT_VERSION or batch.profile != engine.settings.profile:
        raise ValueError("Rollout contract/profile mismatch")
    x, stage, mask = engine.tensors(batch.x, batch.gate_stage)
    count = len(x)
    for name in ["z", "u", "old_log_prob", "old_value", "returns", "advantages"]:
        value = np.asarray(getattr(batch, name))
        shape = (count, 4) if name in {"z", "u"} else (count,)
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"Invalid float32 rollout field: {name}")
    stored_mask = np.asarray(batch.actor_update_mask)
    if stored_mask.dtype != np.bool_ or not np.array_equal(stored_mask, mask.numpy()):
        raise ValueError("Rollout actor mask does not match the stage profile")
    if not np.array_equal(batch.u, np.clip(batch.z, -1, 1)):
        raise ValueError("Rollout u must be the clipped stored latent z")
    z = torch.as_tensor(batch.z, dtype=torch.float64)
    old_log_prob = torch.as_tensor(batch.old_log_prob, dtype=torch.float64)
    returns = torch.as_tensor(batch.returns, dtype=torch.float64)
    advantages = torch.as_tensor(batch.advantages, dtype=torch.float64).clone()
    if int(mask.sum()) > 1:
        selected = advantages[mask]
        advantages[mask] = (selected - selected.mean()) / (selected.std(unbiased=False) + 1e-8)
    with torch.no_grad():
        reference_raw = engine.reference(x)
        reference_u = reference_raw.clamp(-1, 1)
    metrics = {
        "run_id": batch.run_id,
        "profile": batch.profile,
        "samples": count,
        "actor_samples": int(mask.sum()),
        "actor_optimizer_steps": 0,
        "critic_optimizer_steps": 0,
        "early_stopped_kl": False,
        "actor_loss": 0.0,
        "critic_loss": 0.0,
        "anchor_mse": 0.0,
        "raw_reference_mse": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "entropy": 0.0,
        "saturation_fraction": float(np.mean(np.abs(batch.u) >= 1)),
        "execution_changed_fraction": float(
            np.mean(np.any(np.abs(batch.u - np.clip(reference_raw.numpy(), -1, 1)) > 1e-5, axis=1))
        ),
    }
    # A KL stop disables subsequent actor steps but still fits all critic batches.
    stop_actor = False
    for _ in range(engine.settings.epochs):
        order = torch.randperm(count, generator=engine.generator)
        for start in range(0, count, engine.settings.minibatch_size):
            indices = order[start : start + engine.settings.minibatch_size]
            active = mask[indices]
            engine.actor_optimizer.zero_grad(set_to_none=True)
            engine.critic_optimizer.zero_grad(set_to_none=True)
            prediction = engine.critic(x[indices], stage[indices])
            critic_loss = 0.5 * (prediction - returns[indices]).square().mean()
            if not bool(torch.isfinite(critic_loss)):
                raise FloatingPointError("Non-finite critic loss")
            critic_loss.backward()
            critic_norm = torch.nn.utils.clip_grad_norm_(
                engine.critic.parameters(), engine.settings.max_grad_norm, error_if_nonfinite=True
            )
            engine.critic_optimizer.step()
            metrics["critic_optimizer_steps"] += 1
            metrics["critic_loss"] += float(critic_loss.detach())
            metrics["critic_grad_norm_last"] = float(critic_norm)
            if stop_actor or not bool(active.any()):
                continue
            selected = indices[active]
            distribution = engine.distribution(x[selected])
            new_log_prob = distribution.log_prob(z[selected]).sum(dim=1)
            log_ratio = new_log_prob - old_log_prob[selected]
            ratio = log_ratio.exp()
            kl = ((ratio - 1) - log_ratio).mean().clamp_min(0)
            if not bool(torch.isfinite(ratio).all()) or not bool(torch.isfinite(kl)):
                raise FloatingPointError("Non-finite PPO ratio/KL; discard this training run")
            if float(kl.detach()) > engine.settings.target_kl:
                stop_actor = True
                metrics["early_stopped_kl"] = True
                metrics["kl_stop_value"] = float(kl.detach())
                continue
            clipped = ratio.clamp(1 - engine.settings.clip_range, 1 + engine.settings.clip_range)
            surrogate = -torch.minimum(
                ratio * advantages[selected], clipped * advantages[selected]
            ).mean()
            anchor = (distribution.mean.clamp(-1, 1) - reference_u[selected]).square().mean()
            raw_deviation = (distribution.mean - reference_raw[selected]).square().mean()
            entropy = distribution.entropy().sum(dim=1).mean()
            loss = (
                surrogate
                + engine.settings.anchor_coef * anchor
                - engine.settings.entropy_coef * entropy
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite actor loss")
            loss.backward()
            actor_norm = torch.nn.utils.clip_grad_norm_(
                [*engine.actor.parameters(), engine.log_std],
                engine.settings.max_grad_norm,
                error_if_nonfinite=True,
            )
            engine.actor_optimizer.step()
            with torch.no_grad():
                engine.log_std.clamp_(-8, 1)
            metrics["actor_optimizer_steps"] += 1
            for key, value in {
                "actor_loss": loss,
                "anchor_mse": anchor,
                "raw_reference_mse": raw_deviation,
                "approx_kl": kl,
                "clip_fraction": (torch.abs(ratio - 1) > engine.settings.clip_range)
                .double()
                .mean(),
                "entropy": entropy,
            }.items():
                metrics[key] += float(value.detach())
            metrics["actor_grad_norm_last"] = float(actor_norm)
    actor_steps = max(1, metrics["actor_optimizer_steps"])
    for key in [
        "actor_loss",
        "anchor_mse",
        "raw_reference_mse",
        "approx_kl",
        "clip_fraction",
        "entropy",
    ]:
        metrics[key] /= actor_steps
    metrics["critic_loss"] /= metrics["critic_optimizer_steps"]
    engine.update_count += 1
    metrics["update_count"] = engine.update_count
    return metrics
