"""NumPy-only stage policy reader and explicit two-actor deployment export."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.team_policy_runtime import SCHEMA_VERSION, NumpyStagePolicy
from training.team_a.npz_io import ORIGINAL_MODEL, ORIGINAL_SHA256, file_sha256
from training.team_contract import CONTRACT_VERSION

if TYPE_CHECKING:
    from training.team_a.engine import StagePPOEngine


def export_stage_actor(engine: StagePPOEngine, path: Path, manifest: dict) -> dict:
    """Export both actors, normalization and stage rules without Torch deployment code."""
    required = {"run_id", "policy_commit", "training_commit", "training_kind"}
    if not required <= manifest.keys() or not all(manifest[key] for key in required):
        raise ValueError(f"Export requires provenance fields {sorted(required)}")
    if manifest["training_kind"] not in {"baseline-conversion", "synthetic-smoke", "racing"}:
        raise ValueError("training_kind must distinguish baseline, synthetic tests and racing")
    if manifest["training_kind"] == "baseline-conversion":
        unchanged = all(
            np.array_equal(
                value.detach().cpu().numpy(),
                engine.reference.state_dict()[key].detach().cpu().numpy(),
            )
            for key, value in engine.actor.state_dict().items()
        )
        if engine.update_count != 0 or not unchanged:
            raise ValueError("Baseline conversion requires the unchanged, never-updated actor")
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
        "training_kind": np.array(manifest["training_kind"]),
        "update_count": np.array(engine.update_count, dtype=np.int64),
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
