"""NumPy-only stage policy reader and explicit two-actor deployment export."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from training.team_a.npz_io import (
    LAYER_SHAPES,
    ORIGINAL_MODEL,
    ORIGINAL_SHA256,
    file_sha256,
    numpy_mean,
)
from training.team_contract import CONTRACT_VERSION

if TYPE_CHECKING:
    from training.team_a.engine import StagePPOEngine

SCHEMA_VERSION = "stage-ppo-v1"


def export_stage_actor(engine: StagePPOEngine, path: Path, manifest: dict) -> dict:
    """Export both actors, normalization and stage rules without Torch deployment code."""
    required = {"run_id", "policy_commit", "training_commit", "training_kind"}
    if not required <= manifest.keys() or not all(manifest[key] for key in required):
        raise ValueError(f"Export requires provenance fields {sorted(required)}")
    if manifest["training_kind"] not in {"synthetic-smoke", "racing"}:
        raise ValueError("training_kind must distinguish synthetic tests from racing")
    path = Path(path)
    if path.suffix != ".npz" or path.resolve() == ORIGINAL_MODEL.resolve():
        raise ValueError("Choose a new .npz path, never the original baseline")
    arrays = {
        "schema_version": np.array(SCHEMA_VERSION),
        "contract_version": np.array(CONTRACT_VERSION),
        "profile": np.array(engine.settings.profile),
        "original_model_sha256": np.array(ORIGINAL_SHA256),
        "n_stages": np.array(engine.settings.n_stages, dtype=np.int64),
        "active_stages": np.asarray(engine.settings.active_stages, dtype=np.int64),
    }
    for prefix, actor in [("base", engine.reference), ("tuned", engine.actor)]:
        arrays[f"{prefix}_obs_mean"] = actor.obs_mean.detach().cpu().numpy().copy()
        arrays[f"{prefix}_obs_std"] = actor.obs_std.detach().cpu().numpy().copy()
        arrays[f"{prefix}_tilt_max"] = np.array(actor.tilt_max)
        for index, layer in enumerate(actor.layers):
            arrays[f"{prefix}_W{index}"] = layer.weight.detach().cpu().numpy().T.copy()
            arrays[f"{prefix}_b{index}"] = layer.bias.detach().cpu().numpy().copy()
    for name, value in arrays.items():
        if value.dtype.kind != "U" and not np.isfinite(value).all():
            raise ValueError(f"Cannot export non-finite array {name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        np.savez(output, **arrays)
    digest = file_sha256(path)
    # Validate the serialized reader now; never call an untested export deployable.
    NumpyStagePolicy(path, expected_sha256=digest)
    return {
        "schema_version": SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "artifact_sha256": digest,
        "original_model_sha256": ORIGINAL_SHA256,
        "profile": engine.settings.profile,
        "active_stages": list(engine.settings.active_stages),
        "n_stages": engine.settings.n_stages,
        "transition_rule": "hard-switch-on-official-stage",
        "normalization": "original-mean-std-once-clip-10",
        "action_mapping": "roll-pitch-yaw-thrust-v1-tilt-1.5",
        "update_count": engine.update_count,
        "training_manifest": dict(manifest),
        "source": "Xiaolong-Li1904/lsy_drone_racing@1cca34f2476d7791bdb7074fc9cf3dc32b4fbc27",
        "license": "lsy_drone_racing/control/policies/level2_reference_LICENSE",
    }


class NumpyStagePolicy:
    """Deterministic two-actor reader; neither this module nor npz_io imports Torch."""

    def __init__(self, path: Path, expected_sha256: str) -> None:
        """Validate a frozen exported artifact and its stage selection metadata."""
        if file_sha256(path) != expected_sha256:
            raise ValueError("Stage actor SHA256 mismatch")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
        if (
            arrays["schema_version"].item() != SCHEMA_VERSION
            or arrays["contract_version"].item() != CONTRACT_VERSION
            or arrays["original_model_sha256"].item() != ORIGINAL_SHA256
        ):
            raise ValueError("Stage actor schema, contract or original source mismatch")
        self.profile = str(arrays["profile"].item())
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
            if any(not np.isfinite(a).all() for a in actor.values()):
                raise ValueError("Non-finite actor arrays")
            if (
                actor["obs_mean"].shape != (46,)
                or actor["obs_std"].shape != (46,)
                or np.any(actor["obs_std"] <= 0)
                or float(actor["tilt_max"]) != 1.5
            ):
                raise ValueError("Invalid normalization or tilt mapping")
            for i, shape in enumerate(LAYER_SHAPES):
                if actor[f"W{i}"].shape != shape or actor[f"b{i}"].shape != (shape[1],):
                    raise ValueError("Invalid layer shape")
            self.actors[prefix] = actor
        for name in ["obs_mean", "obs_std", "tilt_max"]:
            if not np.array_equal(self.actors["base"][name], self.actors["tuned"][name]):
                raise ValueError("Frozen normalization must match across actors")

    def commands(self, x: np.ndarray, gate_stage: np.ndarray) -> np.ndarray:
        """Return final normalized commands with the same profile as training."""
        stage = np.asarray(gate_stage)
        if (
            stage.shape != (len(x),)
            or stage.dtype != np.int64
            or np.any(stage < 0)
            or np.any(stage >= self.n_stages)
        ):
            raise ValueError("Invalid gate_stage")
        base = numpy_mean(x, self.actors["base"])
        tuned = numpy_mean(x, self.actors["tuned"])
        mask = np.isin(stage, self.active_stages)
        return np.clip(np.where(mask[:, None], tuned, base), -1, 1).astype(np.float32)
