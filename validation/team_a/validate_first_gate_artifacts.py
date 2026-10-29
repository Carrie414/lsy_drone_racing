"""A-owned deployment audit of first-gate models on recorded public states."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from lsy_drone_racing.control.level2_reference import reference_features
from lsy_drone_racing.control.team_policy_runtime import NumpyStagePolicy
from lsy_drone_racing.envs.race_core import build_action_space
from lsy_drone_racing.utils import load_config
from training.team_a.engine import StagePPOEngine
from training.team_a.policy import from_environment_action
from training.team_a.train_first_gate import first_gate_settings

ROOT = Path(__file__).resolve().parents[2]


def recorded_features(trace_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Select the first and last state of each observed stage in each episode."""
    config = load_config(ROOT / "config/level2.toml")
    space = build_action_space("attitude", config.sim.drone, config.sim.dynamics)
    low, high = float(space.low[3]), float(space.high[3])
    features, stages = [], []
    for path in sorted(trace_dir.glob("trace-*.jsonl.gz")):
        previous = np.zeros(4, dtype=np.float32)
        selected = {}
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                obs = {key: np.asarray(value) for key, value in row["obs"].items()}
                stage = int(row["gate_stage"])
                x = reference_features(obs, previous)
                selected.setdefault((stage, "first"), x)
                selected[(stage, "last")] = x
                previous = from_environment_action(np.asarray([row["action"]]), low, high)[0]
        for (stage, _), x in selected.items():
            features.append(x)
            stages.append(stage)
    if not features:
        raise ValueError("Recorded public trajectories are required")
    return np.asarray(features, dtype=np.float32), np.asarray(stages, dtype=np.int64)


def audit(folder: Path, x: np.ndarray, stage: np.ndarray) -> dict:
    """Check the actual checkpoint, export and frozen base, never report race scores."""
    state = torch.load(folder / "checkpoint.pt", weights_only=True)
    settings = first_gate_settings(int(state["engine"]["settings"]["seed"]))
    engine = StagePPOEngine(settings)
    engine.load_training_state_dict(state["engine"])
    manifest = json.loads((folder / "candidate.manifest.json").read_text(encoding="utf-8"))
    policy = NumpyStagePolicy(folder / "candidate.npz", manifest["artifact_sha256"])
    if policy.active_stages.tolist() != [0] or policy.profile != settings.profile:
        raise ValueError("Wrong first-gate profile")
    reference = engine.reference.deterministic(x)
    torch_result = engine.act(x, stage, deterministic=True).u
    numpy_result = policy.commands(x, stage)
    max_error = float(np.max(np.abs(torch_result - numpy_result)))
    np.testing.assert_allclose(torch_result, numpy_result, atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(torch_result[stage != 0], reference[stage != 0])
    correction = np.abs(numpy_result[stage == 0] - reference[stage == 0])
    assert np.all(correction <= np.array(settings.residual_bounds) + 1e-6)
    assert all(
        torch.equal(value, engine.actor.base.state_dict()[key])
        for key, value in engine.reference.state_dict().items()
    )
    return {
        "artifact": str(folder),
        "artifact_sha256": manifest["artifact_sha256"],
        "rows": len(x),
        "stage_counts": {str(s): int(np.sum(stage == s)) for s in range(6)},
        "active_stages": [0],
        "update_count": engine.update_count,
        "numpy_torch_max_normalized_action_error": max_error,
        "max_correction_per_axis": correction.max(axis=0).tolist(),
        "frozen_reference_exact": True,
        "inactive_stage_fallback_exact_in_torch": True,
        "note": "Compatibility audit on C0 public states; not on-policy performance evaluation.",
    }


def main() -> None:
    """Audit all four preregistered artifacts and write a local JSON record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    x, stage = recorded_features(args.trace_dir)
    results = [
        audit(args.output_root / f"A4-{seed}-u{updates}", x, stage)
        for seed in (20261029, 20261030)
        for updates in (32, 64)
    ]
    (args.output_root / "deployment_audit.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
