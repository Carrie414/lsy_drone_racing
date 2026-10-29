"""Bounded residual actor for C3, leaving the original racing actor frozen."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from training.team_a.policy import PretrainedActor


class BoundedResidualActor(nn.Module):
    """Predict only a bounded correction around the clipped original command."""

    def __init__(self, reference: PretrainedActor, bounds: tuple[float, ...]) -> None:
        """Zero-initialize the correction to preserve the initial deterministic policy."""
        super().__init__()
        self.base = copy.deepcopy(reference).requires_grad_(False).eval()
        self.register_buffer("obs_mean", reference.obs_mean.detach().clone())
        self.register_buffer("obs_std", reference.obs_std.detach().clone())
        self.register_buffer("bounds", torch.tensor(bounds, dtype=torch.float64))
        self.tilt_max = reference.tilt_max
        self.network = nn.Sequential(
            nn.Linear(46, 64, dtype=torch.float64),
            nn.ELU(),
            nn.Linear(64, 64, dtype=torch.float64),
            nn.ELU(),
            nn.Linear(64, 4, dtype=torch.float64),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the bounded corrected mean, before Gaussian exploration/clipping."""
        value = x.to(dtype=self.obs_mean.dtype, device=self.obs_mean.device)
        normalized = ((value - self.obs_mean) / self.obs_std).clamp(-10, 10)
        baseline = self.base(value).clamp(-1, 1)
        return baseline + self.bounds * self.network(normalized).tanh()
