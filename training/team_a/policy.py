"""A0: import the pretrained actor into Torch without changing its behavior.

This module provides conversion and deterministic inference, not PPO training.
Original actor and MIT provenance are documented in npz_io.py and the baseline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from lsy_drone_racing.control.level2_reference import reference_features
from training.team_a.npz_io import ORIGINAL_MODEL, ORIGINAL_SHA256, load_actor_arrays

if TYPE_CHECKING:
    from pathlib import Path


def encode_obs(batch_obs: list[dict], previous_u: np.ndarray) -> np.ndarray:
    """Use the original encoder exactly once, with one history per environment."""
    previous = np.asarray(previous_u, dtype=np.float32)
    if previous.shape != (len(batch_obs), 4) or not np.isfinite(previous).all():
        raise ValueError("previous_u must be finite [B, 4]")
    if np.any(np.abs(previous) > 1):
        raise ValueError("previous_u must contain final normalized commands in [-1, 1]")
    if not batch_obs:
        raise ValueError("At least one public observation is required")
    features = np.stack([reference_features(obs, previous[i]) for i, obs in enumerate(batch_obs)])
    if features.shape != (len(batch_obs), 46) or not np.isfinite(features).all():
        raise ValueError("Public observations produced invalid [B, 46] features")
    return features


def to_environment_action(
    u: np.ndarray, thrust_low: float, thrust_high: float, tilt_max: float = 1.5
) -> np.ndarray:
    """Map normalized commands to roll/pitch/yaw radians and thrust newtons."""
    value = np.asarray(u, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 4 or not np.isfinite(value).all():
        raise ValueError("Commands must be finite [B, 4]")
    if not np.isfinite([thrust_low, thrust_high, tilt_max]).all():
        raise ValueError("Action mapping constants must be finite")
    if thrust_high <= thrust_low or tilt_max <= 0:
        raise ValueError("Invalid action mapping bounds")
    result = np.clip(value, -1.0, 1.0).copy()
    result[:, :2] *= tilt_max
    result[:, 2] *= 0.5
    result[:, 3] = thrust_low + 0.5 * (result[:, 3] + 1) * (thrust_high - thrust_low)
    return result.astype(np.float32)


def from_environment_action(
    action: np.ndarray, thrust_low: float, thrust_high: float, tilt_max: float = 1.5
) -> np.ndarray:
    """Recover history from the final command, before environment action noise."""
    # Validate dimensions/constants using the forward mapping's common checks.
    to_environment_action(action, thrust_low, thrust_high, tilt_max)
    result = np.asarray(action, dtype=np.float64).copy()
    result[:, :2] /= tilt_max
    result[:, 2] /= 0.5
    result[:, 3] = 2 * (result[:, 3] - thrust_low) / (thrust_high - thrust_low) - 1
    return np.clip(result, -1.0, 1.0).astype(np.float32)


class PretrainedActor(nn.Module):
    """Differentiable 46-256-256-256-4 ELU actor with original normalization.

    Float64 defaults to the deployed baseline's arithmetic for A0 compatibility.
    Selecting float32 for later training requires passing its own compatibility
    tests. There is no critic, exploration distribution or optimizer in A0.
    """

    def __init__(
        self,
        path: Path = ORIGINAL_MODEL,
        expected_sha256: str | None = ORIGINAL_SHA256,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        """Load the original weights, transposing each NumPy matrix for Linear."""
        super().__init__()
        arrays = load_actor_arrays(path, expected_sha256)
        self.register_buffer("obs_mean", torch.tensor(arrays["obs_mean"], dtype=dtype))
        self.register_buffer("obs_std", torch.tensor(arrays["obs_std"], dtype=dtype))
        self.tilt_max = float(arrays["tilt_max"])
        self.layers = nn.ModuleList()
        for index in range(4):
            weight, bias = arrays[f"W{index}"], arrays[f"b{index}"]
            layer = nn.Linear(weight.shape[0], weight.shape[1], dtype=dtype)
            with torch.no_grad():
                layer.weight.copy_(torch.as_tensor(weight.T.copy(), dtype=dtype))
                layer.bias.copy_(torch.as_tensor(bias, dtype=dtype))
            self.layers.append(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the raw linear mean; the pretrained actor has no output tanh."""
        if x.ndim != 2 or x.shape[1] != 46 or not bool(torch.isfinite(x).all()):
            raise ValueError("Features must be finite [B, 46]")
        value = x.to(device=self.obs_mean.device, dtype=self.obs_mean.dtype)
        value = torch.clamp((value - self.obs_mean) / self.obs_std, -10, 10)
        for index, layer in enumerate(self.layers):
            value = layer(value)
            if index != 3:
                value = torch.nn.functional.elu(value)
        return value

    def deterministic(self, x: np.ndarray) -> np.ndarray:
        """Return clipped normalized commands using NumPy at the API boundary."""
        with torch.no_grad():
            value = self(torch.as_tensor(x, device=self.obs_mean.device))
        return value.clamp(-1, 1).cpu().numpy().astype(np.float32)
