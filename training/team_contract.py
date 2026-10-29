"""Version 2 NumPy boundary shared by team A policies and team B runners.

The records and protocol define the agreed boundary. A0 supplies actor conversion;
A1's StagePPOEngine supplies stochastic action selection, a critic and PPO updates.
Team B owns racing rewards, complete rollouts and GAE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

CONTRACT_VERSION = "v2"
FEATURE_DIM = 46
ACTION_DIM = 4


@dataclass(frozen=True)
class PolicyStep:
    """One batched engine result, with Gaussian density referring to unclipped z."""

    mu: np.ndarray
    z: np.ndarray
    u: np.ndarray
    log_prob: np.ndarray
    value: np.ndarray
    actor_update_mask: np.ndarray


@dataclass(frozen=True)
class RolloutBatch:
    """Flattened rollout with returns already computed by the team B runner."""

    x: np.ndarray
    z: np.ndarray
    u: np.ndarray
    old_log_prob: np.ndarray
    old_value: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray
    gate_stage: np.ndarray
    actor_update_mask: np.ndarray
    run_id: str
    profile: str
    contract_version: str = CONTRACT_VERSION


class PolicyEngine(Protocol):
    """Version 2 boundary implemented by team_a.engine.StagePPOEngine."""

    def encode_obs(self, batch_obs: list[dict], previous_u: np.ndarray) -> np.ndarray:
        """Return unnormalized [B, 46] public features."""
        ...

    def act(self, x: np.ndarray, gate_stage: np.ndarray, deterministic: bool = False) -> PolicyStep:
        """Return commands, density, stage-aware value and actor mask."""
        ...

    def value(self, x: np.ndarray, gate_stage: np.ndarray) -> np.ndarray:
        """Evaluate remaining return with the explicit gate sequence stage."""
        ...

    def update(self, batch: RolloutBatch) -> dict:
        """Apply the team A updater to a team B rollout."""
        ...

    def training_state_dict(self) -> dict:
        """Return policy, critic, exploration and optimizer state."""
        ...

    def load_training_state_dict(self, state: dict) -> None:
        """Restore the training components owned by team A."""
        ...

    def export_npz(self, path: Path, manifest: dict) -> dict:
        """Export deterministic deployment arrays and provenance."""
        ...
