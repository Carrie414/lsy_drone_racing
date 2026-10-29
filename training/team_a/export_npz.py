"""Export an A0 actor to the original NumPy layout with explicit provenance."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

import numpy as np

from training.team_a.npz_io import ORIGINAL_MODEL, ORIGINAL_SHA256, file_sha256
from training.team_contract import CONTRACT_VERSION

if TYPE_CHECKING:
    from training.team_a.policy import PretrainedActor


def export_actor(actor: PretrainedActor, path: Path) -> dict:
    """Write a new archive, rejecting overwrites and preserving the baseline asset."""
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Export path must end in .npz")
    if path.resolve() == ORIGINAL_MODEL.resolve():
        raise ValueError("The original baseline model must never be overwritten")
    if path.exists():
        raise FileExistsError("Choose a new artifact path; exports are immutable")
    arrays = {
        "obs_mean": actor.obs_mean.detach().cpu().numpy().copy(),
        "obs_std": actor.obs_std.detach().cpu().numpy().copy(),
        "n_layers": np.array(len(actor.layers)),
        "tilt_max": np.array(actor.tilt_max),
    }
    for index, layer in enumerate(actor.layers):
        arrays[f"W{index}"] = layer.weight.detach().cpu().numpy().T.copy()
        arrays[f"b{index}"] = layer.bias.detach().cpu().numpy().copy()
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Cannot export non-finite actor arrays")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents accidental overwrite if another export races us.
    with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as temporary:
        temporary_path = Path(temporary.name)
        np.savez(temporary, **arrays)
    try:
        with path.open("xb") as output, temporary_path.open("rb") as source:
            while block := source.read(1024 * 1024):
                output.write(block)
    finally:
        os.unlink(temporary_path)
    return {
        "contract_version": CONTRACT_VERSION,
        "artifact_sha256": file_sha256(path),
        "original_model_sha256": ORIGINAL_SHA256,
        "stage": "A0-conversion-only-not-trained",
        "architecture": [46, 256, 256, 256, 4],
        "tilt_max": actor.tilt_max,
        "dtype": str(actor.obs_mean.dtype),
        "source": "Xiaolong-Li1904/lsy_drone_racing@1cca34f2476d7791bdb7074fc9cf3dc32b4fbc27",
        "license": "lsy_drone_racing/control/policies/level2_reference_LICENSE",
    }
