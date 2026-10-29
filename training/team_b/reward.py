"""Training-only racing reward from official progress and public observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RewardWeights:
    """Initial smoke-test recipe, not tuned or proven performance parameters."""

    progress: float = 1.0
    gate: float = 5.0
    finish: float = 20.0
    time: float = 1.0
    failure: float = 5.0
    action_change: float = 0.0
    max_progress_step: float = 0.5


def racing_reward(
    previous: dict,
    current: dict,
    u: np.ndarray,
    previous_u: np.ndarray,
    *,
    terminated: bool = False,
    truncated: bool = False,
    dt: float = 0.02,
    weights: RewardWeights = RewardWeights(),
) -> tuple[float, dict]:
    """Return (reward, components); caller supplies final obs before any autoreset."""
    stage = int(previous["n_gates_passed"])
    after = int(current["n_gates_passed"])
    count = len(previous["gate_sequence"])
    if after < stage or not 0 <= stage <= after <= count:
        raise ValueError("Non-monotone gate count: use terminal final observation before reset")
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError("Invalid dt")
    success = after == count
    completed_now = stage < count and success
    failure = bool((terminated or truncated) and not success)
    progress = 0.0
    if stage < count and not failure:
        # Hold the PRE-step observed target fixed: discovery/target switches earn no fake progress.
        gate = int(previous["gate_sequence"][stage])
        goal = np.asarray(previous["gates_pos"][gate], dtype=float)
        progress = float(
            np.linalg.norm(np.asarray(previous["pos"]) - goal)
            - np.linalg.norm(np.asarray(current["pos"]) - goal)
        )
        progress = float(np.clip(progress, -weights.max_progress_step, weights.max_progress_step))
    change = float(np.sum((np.asarray(u) - np.asarray(previous_u)) ** 2))
    parts = dict(
        progress=weights.progress * progress,
        gate=weights.gate * (after - stage),
        finish=weights.finish * completed_now,
        time=-weights.time * dt,
        failure=-weights.failure * failure,
        action_change=-weights.action_change * change,
    )
    reward = float(sum(parts.values()))
    if not np.isfinite(reward):
        raise ValueError("Non-finite reward")
    return reward, parts
