"""NumPy-only actor archive validation and forward evaluation.

Original actor: Xiaolong-Li1904/lsy_drone_racing at
1cca34f2476d7791bdb7074fc9cf3dc32b4fbc27 (MIT).
The original license remains in control/policies/level2_reference_LICENSE.
No Torch import is permitted in this deployment-compatible module.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

ORIGINAL_SHA256 = "d15e6d613041907ed5baa565f87ee3393df099b78ecb53b4428440399baede9a"
ORIGINAL_MODEL = (
    Path(__file__).resolve().parents[2] / "lsy_drone_racing/control/policies/level2_reference.npz"
)
LAYER_SHAPES = ((46, 256), (256, 256), (256, 256), (256, 4))


def file_sha256(path: Path) -> str:
    """Hash a selected model without unpickling it."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_actor_arrays(path: Path, expected_sha256: str | None = None) -> dict:
    """Validate the exact original architecture and positive normalization scale."""
    path = Path(path)
    if expected_sha256 is not None and file_sha256(path) != expected_sha256:
        raise ValueError("Actor SHA256 does not match the selected artifact")
    with np.load(path, allow_pickle=False) as archive:
        names = ["obs_mean", "obs_std", "n_layers", "tilt_max"]
        names += [f"{part}{i}" for i in range(4) for part in ("W", "b")]
        missing = set(names) - set(archive.files)
        if missing:
            raise ValueError(f"Actor archive missing arrays: {sorted(missing)}")
        arrays = {name: np.array(archive[name], copy=True) for name in names}
    for name, value in arrays.items():
        if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError(f"Invalid actor array: {name}")
    if arrays["n_layers"].shape != () or float(arrays["n_layers"]) != 4:
        raise ValueError("Actor must contain four linear layers")
    if arrays["obs_mean"].shape != (46,) or arrays["obs_std"].shape != (46,):
        raise ValueError("Actor normalization must have 46 features")
    if np.any(arrays["obs_std"] <= 0):
        raise ValueError("Actor standard deviations must be positive")
    if arrays["tilt_max"].shape != () or float(arrays["tilt_max"]) != 1.5:
        raise ValueError("This Level 2 profile requires tilt_max=1.5")
    for index, shape in enumerate(LAYER_SHAPES):
        if arrays[f"W{index}"].shape != shape or arrays[f"b{index}"].shape != (shape[1],):
            raise ValueError(f"Invalid actor layer {index} dimensions")
    return arrays


def numpy_mean(x: np.ndarray, arrays: dict) -> np.ndarray:
    """Evaluate the unclipped mean with the original float64 NumPy arithmetic."""
    value = np.asarray(x, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 46 or not np.isfinite(value).all():
        raise ValueError("Features must be finite [B, 46] arrays")
    value = np.clip((value - arrays["obs_mean"]) / arrays["obs_std"], -10.0, 10.0)
    for index in range(4):
        value = value @ arrays[f"W{index}"].astype(np.float64)
        value += arrays[f"b{index}"].astype(np.float64)
        if index != 3:
            value = np.where(value > 0, value, np.expm1(np.minimum(value, 0)))
    return value
