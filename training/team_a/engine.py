"""Stage-aware PPO engine using public features and the frozen Level 2 actor."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from training.team_a.npz_io import ORIGINAL_SHA256
from training.team_a.policy import PretrainedActor, encode_obs
from training.team_a.ppo_update import update_engine
from training.team_contract import CONTRACT_VERSION, PolicyStep

if TYPE_CHECKING:
    from pathlib import Path

    from training.team_contract import RolloutBatch

ENGINE_VERSION = "stage-ppo-v1"


@dataclass(frozen=True)
class PPOSettings:
    """Small initial settings, not empirically optimized racing hyperparameters."""

    seed: int = 20261029
    profile: str = "C1-stage-ppo-v1"
    n_stages: int = 6  # Five official passes plus the completed stage.
    active_stages: tuple[int, ...] = (0, 2)
    actor_lr: float = 1e-5
    critic_lr: float = 1e-3
    initial_std: float = 0.05
    clip_range: float = 0.2
    anchor_coef: float = 1.0
    entropy_coef: float = 0.0
    max_grad_norm: float = 0.5
    target_kl: float = 0.02
    epochs: int = 4
    minibatch_size: int = 64

    def __post_init__(self) -> None:
        """Reject invalid settings before optimizer creation."""
        if self.n_stages < 2 or not self.active_stages:
            raise ValueError("At least two stages and one active stage are required")
        if len(set(self.active_stages)) != len(self.active_stages):
            raise ValueError("Active stages must be unique")
        if any(type(s) is not int or s < 0 or s >= self.n_stages for s in self.active_stages):
            raise ValueError("Active stage outside the declared range")
        positive = [
            self.actor_lr,
            self.critic_lr,
            self.initial_std,
            self.clip_range,
            self.max_grad_norm,
            self.target_kl,
        ]
        if not np.isfinite(positive).all() or min(positive) <= 0:
            raise ValueError("Learning rates, exploration and PPO limits must be positive")
        if self.clip_range >= 1 or not self.profile:
            raise ValueError("Invalid PPO clipping range or profile")
        if not np.isfinite([self.anchor_coef, self.entropy_coef]).all():
            raise ValueError("Loss weights must be finite")
        if min(self.anchor_coef, self.entropy_coef) < 0:
            raise ValueError("Loss weights must be nonnegative")
        if self.epochs < 1 or self.minibatch_size < 1:
            raise ValueError("Epochs and minibatch_size must be positive integers")


class StageCritic(nn.Module):
    """Value function with frozen normalization and explicit stage one-hot input."""

    def __init__(self, reference: PretrainedActor, n_stages: int) -> None:
        """Construct a new critic; the original NPZ has no critic checkpoint."""
        super().__init__()
        self.n_stages = n_stages
        self.register_buffer("obs_mean", reference.obs_mean.detach().clone())
        self.register_buffer("obs_std", reference.obs_std.detach().clone())
        self.network = nn.Sequential(
            nn.Linear(46 + n_stages, 128, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(128, 128, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(128, 1, dtype=torch.float64),
        )

    def forward(self, x: torch.Tensor, stage: torch.Tensor) -> torch.Tensor:
        """Predict return while distinguishing repeated passes through a gate."""
        normalized = ((x - self.obs_mean) / self.obs_std).clamp(-10, 10)
        one_hot = torch.nn.functional.one_hot(stage, self.n_stages).to(normalized.dtype)
        return self.network(torch.cat([normalized, one_hot], dim=1)).squeeze(-1)


class StagePPOEngine:
    """Implement the v2 engine with independent actor and critic optimizers.

    Stage selection is a hard, explicit function of the official stage. The
    runner owns per-environment command history, rewards, GAE and environment RNG.
    The engine does not read simulator state or run environment rollouts.
    """

    def __init__(self, settings: PPOSettings | None = None) -> None:
        """Start a CPU float64 engine without modifying the caller's Torch RNG."""
        self.settings = settings or PPOSettings()
        # This first implementation deliberately targets the verified CPU setup.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.settings.seed)
            self.reference = PretrainedActor().requires_grad_(False).eval()
            self.actor = copy.deepcopy(self.reference).requires_grad_(True)
            self.critic = StageCritic(self.reference, self.settings.n_stages)
        self.log_std = nn.Parameter(
            torch.full((4,), np.log(self.settings.initial_std), dtype=torch.float64)
        )
        self.actor_optimizer = torch.optim.Adam(
            [*self.actor.parameters(), self.log_std], lr=self.settings.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.settings.critic_lr
        )
        self.generator = torch.Generator(device="cpu").manual_seed(self.settings.seed)
        self.update_count = 0

    @staticmethod
    def encode_obs(batch_obs: list[dict], previous_u: np.ndarray) -> np.ndarray:
        """Return the original unnormalized public encoding."""
        return encode_obs(batch_obs, previous_u)

    def tensors(
        self, x: np.ndarray, gate_stage: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Validate the NumPy boundary and derive the single source of stage masks."""
        features, stages = np.asarray(x), np.asarray(gate_stage)
        if features.ndim != 2 or features.shape[1] != 46 or not len(features):
            raise ValueError("x must have nonempty [B, 46] shape")
        if features.dtype != np.float32 or not np.isfinite(features).all():
            raise ValueError("x must be finite unnormalized float32 features")
        if np.any(np.abs(features[:, 42:46]) > 1 + 1e-6):
            raise ValueError("Feature history must contain final commands in [-1, 1]")
        if stages.shape != (len(features),) or stages.dtype != np.int64:
            raise ValueError("gate_stage must be int64 [B]")
        if np.any(stages < 0) or np.any(stages >= self.settings.n_stages):
            raise ValueError("gate_stage outside the declared range")
        mask = np.isin(stages, self.settings.active_stages)
        return (
            torch.as_tensor(features, dtype=torch.float64),
            torch.as_tensor(stages, dtype=torch.int64),
            torch.as_tensor(mask, dtype=torch.bool),
        )

    def distribution(self, x: torch.Tensor) -> Normal:
        """Return the trainable pre-clipping Gaussian, never density of clipped u."""
        return Normal(self.actor(x), self.log_std.clamp(-8, 1).exp())

    def act(self, x: np.ndarray, gate_stage: np.ndarray, deterministic: bool = False) -> PolicyStep:
        """Sample only trainable stages; inactive stages execute the frozen actor."""
        features, stages, mask = self.tensors(x, gate_stage)
        with torch.no_grad():
            baseline = self.reference(features)
            distribution = self.distribution(features)
            mean = torch.where(mask[:, None], distribution.mean, baseline)
            latent = mean.clone()
            if not deterministic and bool(mask.any()):
                noise = torch.randn(
                    (int(mask.sum()), 4), generator=self.generator, dtype=torch.float64
                )
                latent[mask] += noise * distribution.scale[mask]
            # Store z as float32; density must refer to that same stored value.
            stored_z = latent.float()
            log_prob = distribution.log_prob(stored_z.double()).sum(dim=1)
            log_prob = torch.where(mask, log_prob, torch.zeros_like(log_prob))
            values = self.critic(features, stages)
        return PolicyStep(
            mu=mean.float().numpy(),
            z=stored_z.numpy(),
            u=stored_z.clamp(-1, 1).numpy(),
            log_prob=log_prob.float().numpy(),
            value=values.float().numpy(),
            actor_update_mask=mask.numpy(),
        )

    def value(self, x: np.ndarray, gate_stage: np.ndarray) -> np.ndarray:
        """Evaluate the identical critic used by act, including explicit stage."""
        features, stages, _ = self.tensors(x, gate_stage)
        with torch.no_grad():
            return self.critic(features, stages).float().numpy()

    def update(self, batch: RolloutBatch) -> dict:
        """Update from already-computed team B returns; never recalculate GAE."""
        return update_engine(self, batch)

    def training_state_dict(self) -> dict:
        """Snapshot all A-owned training and sampling state without aliasing tensors."""
        return copy.deepcopy(
            {
                "engine_version": ENGINE_VERSION,
                "contract_version": CONTRACT_VERSION,
                "original_model_sha256": ORIGINAL_SHA256,
                "settings": asdict(self.settings),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "log_std": self.log_std.detach(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "generator_state": self.generator.get_state(),
                "update_count": self.update_count,
            }
        )

    def load_training_state_dict(self, state: dict) -> None:
        """Restore compatible state; caller must supply a trusted local checkpoint."""
        if (
            state.get("engine_version") != ENGINE_VERSION
            or state.get("contract_version") != CONTRACT_VERSION
            or state.get("original_model_sha256") != ORIGINAL_SHA256
            or state.get("settings") != asdict(self.settings)
        ):
            raise ValueError("Checkpoint settings, baseline or protocol mismatch")
        self.actor.load_state_dict(state["actor"], strict=True)
        self.critic.load_state_dict(state["critic"], strict=True)
        with torch.no_grad():
            self.log_std.copy_(state["log_std"])
        self.actor_optimizer.load_state_dict(copy.deepcopy(state["actor_optimizer"]))
        self.critic_optimizer.load_state_dict(copy.deepcopy(state["critic_optimizer"]))
        self.generator.set_state(state["generator_state"])
        self.update_count = int(state["update_count"])

    def export_npz(self, path: Path, manifest: dict) -> dict:
        """Export both actors and the exact hard stage selection for NumPy inference."""
        from training.team_a.stage_export import export_stage_actor

        return export_stage_actor(self, path, manifest)
