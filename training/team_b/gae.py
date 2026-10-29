"""V2 GAE: task terminal/timeout stop bootstrap; a rollout cut does not."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def generalized_advantages(
    rewards: ArrayLike,
    values: ArrayLike,
    next_values: ArrayLike,
    terminated: ArrayLike,
    task_timeouts: ArrayLike,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Inputs [T,N]; next_values use true final obs, not a following episode's reset obs.

    Task deadlines are failures by protocol v2. Sampling cuts are represented by
    neither terminal nor timeout, so the last genuine next value is bootstrapped.
    Stage changes are deliberately not an input and do not stop propagation.
    """
    r = np.asarray(rewards, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    nv = np.asarray(next_values, dtype=np.float64)
    term = np.asarray(terminated, dtype=bool)
    timeout = np.asarray(task_timeouts, dtype=bool)
    if r.ndim != 2 or any(a.shape != r.shape for a in (v, nv, term, timeout)):
        raise ValueError("All rollout arrays must have identical [T,N] shapes")
    if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("Invalid discount")
    if not all(np.isfinite(a).all() for a in (r, v, nv)):
        raise ValueError("Non-finite rollout")
    mask = ~(term | timeout)
    adv = np.empty_like(r)
    carry = np.zeros(r.shape[1])
    for t in range(len(r) - 1, -1, -1):
        delta = r[t] + gamma * mask[t] * nv[t] - v[t]
        carry = delta + gamma * gae_lambda * mask[t] * carry
        adv[t] = carry
    return adv.astype(np.float32), (adv + v).astype(np.float32)
