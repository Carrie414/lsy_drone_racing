"""NumPy-only runtime for exported team actors; no training package imports.

The base model is attributed to Xiaolong-Li1904/lsy_drone_racing at
1cca34f2476d7791bdb7074fc9cf3dc32b4fbc27. Its MIT notice is preserved in
policies_team/level2_reference_LICENSE. The gate-stage profile is team code.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "stage-ppo-v1"
SOURCE_SHA256 = "d15e6d613041907ed5baa565f87ee3393df099b78ecb53b4428440399baede9a"
LAYER_SHAPES = ((46, 256), (256, 256), (256, 256), (256, 4))


def _forward(x: np.ndarray, actor: dict) -> np.ndarray:
    """Evaluate one original-layout actor using frozen normalization."""
    value = np.clip((x - actor["obs_mean"]) / actor["obs_std"], -10, 10)
    for index in range(4):
        value = value @ actor[f"W{index}"] + actor[f"b{index}"]
        if index != 3:
            value = np.where(value > 0, value, np.expm1(np.minimum(value, 0)))
    return value


class NumpyStagePolicy:
    """Read a hash-pinned two-actor artifact without importing Torch or training."""

    def __init__(self, path: Path, expected_sha256: str) -> None:
        """Validate model structure, normalization, source and stage metadata."""
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected_sha256:
            raise ValueError("Stage actor SHA256 mismatch")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
        if (
            arrays["schema_version"].item() != SCHEMA_VERSION
            or arrays["contract_version"].item() != "v2"
            or arrays["original_model_sha256"].item() != SOURCE_SHA256
        ):
            raise ValueError("Stage actor schema, contract or original source mismatch")
        self.profile = str(arrays["profile"].item())
        self.training_kind = str(arrays.get("training_kind", np.array("unknown")).item())
        self.update_count = int(arrays.get("update_count", np.array(-1)))
        if arrays["n_stages"].shape != () or arrays["n_stages"].dtype != np.int64:
            raise ValueError("Invalid stage count")
        self.n_stages = int(arrays["n_stages"])
        self.active_stages = arrays["active_stages"]
        if (
            self.n_stages < 2
            or self.active_stages.ndim != 1
            or self.active_stages.dtype != np.int64
            or not len(self.active_stages)
            or len(np.unique(self.active_stages)) != len(self.active_stages)
            or np.any(self.active_stages < 0)
            or np.any(self.active_stages >= self.n_stages)
        ):
            raise ValueError("Invalid stage selection metadata")
        self.actors = {}
        for prefix in ["base", "tuned"]:
            names = ["obs_mean", "obs_std", "tilt_max"]
            names += [f"{kind}{i}" for i in range(4) for kind in ["W", "b"]]
            actor = {key: arrays[f"{prefix}_{key}"] for key in names}
            if any(a.dtype.kind not in "fiu" or not np.isfinite(a).all() for a in actor.values()):
                raise ValueError("Non-finite or nonnumeric actor arrays")
            if (
                actor["obs_mean"].shape != (46,)
                or actor["obs_std"].shape != (46,)
                or np.any(actor["obs_std"] <= 0)
                or actor["tilt_max"].shape != ()
                or float(actor["tilt_max"]) != 1.5
            ):
                raise ValueError("Invalid normalization or tilt mapping")
            for index, shape in enumerate(LAYER_SHAPES):
                if actor[f"W{index}"].shape != shape or actor[f"b{index}"].shape != (shape[1],):
                    raise ValueError("Invalid layer shape")
            self.actors[prefix] = {name: a.astype(np.float64) for name, a in actor.items()}
        for name in ["obs_mean", "obs_std", "tilt_max"]:
            if not np.array_equal(self.actors["base"][name], self.actors["tuned"][name]):
                raise ValueError("Frozen normalization must match across actors")

    def commands(self, x: np.ndarray, gate_stage: np.ndarray) -> np.ndarray:
        """Return normalized commands, evaluating only each row's selected actor."""
        features, stages = np.asarray(x, dtype=np.float64), np.asarray(gate_stage)
        if features.ndim != 2 or features.shape[1] != 46 or not np.isfinite(features).all():
            raise ValueError("Features must be finite [B, 46]")
        if np.any(np.abs(features[:, 42:46]) > 1 + 1e-6):
            raise ValueError("Feature history must be normalized final commands")
        if (
            stages.shape != (len(features),)
            or stages.dtype != np.int64
            or np.any(stages < 0)
            or np.any(stages >= self.n_stages)
        ):
            raise ValueError("Invalid gate_stage")
        mask = np.isin(stages, self.active_stages)
        result = np.empty((len(features), 4), dtype=np.float64)
        if np.any(mask):
            result[mask] = _forward(features[mask], self.actors["tuned"])
        if np.any(~mask):
            result[~mask] = _forward(features[~mask], self.actors["base"])
        if not np.isfinite(result).all():
            raise ValueError("Actor produced non-finite commands")
        return np.clip(result, -1, 1).astype(np.float32)
