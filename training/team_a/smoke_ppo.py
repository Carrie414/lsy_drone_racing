"""Run a deliberately synthetic A1 smoke check, never a racing performance claim."""

from __future__ import annotations

import argparse
import json
import tempfile
import uuid
from pathlib import Path

import numpy as np
import torch

from training.team_a.engine import PPOSettings, StagePPOEngine
from training.team_a.npz_io import ORIGINAL_MODEL, ORIGINAL_SHA256, file_sha256
from training.team_a.stage_export import NumpyStagePolicy
from training.team_contract import RolloutBatch


def run_smoke(output: Path) -> dict:
    """Exercise sampling, one optimizer update, recovery and deterministic export."""
    if output.exists():
        raise FileExistsError("Use a new output directory to preserve previous evidence")
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    settings = PPOSettings(epochs=1, minibatch_size=16)
    engine = StagePPOEngine(settings)
    rng = np.random.default_rng(75123)
    x = (
        engine.reference.obs_mean.numpy()
        + 0.1 * rng.normal(size=(32, 46)) * engine.reference.obs_std.numpy()
    ).astype(np.float32)
    x[:, 42:] = 0
    stages = np.array([0, 1, 2, 3] * 8, dtype=np.int64)
    deterministic = engine.act(x, stages, deterministic=True)
    reference = engine.reference.deterministic(x)
    np.testing.assert_allclose(deterministic.u, reference, rtol=1e-5, atol=1e-5)
    step = engine.act(x, stages)
    returns = (step.value + np.linspace(-0.4, 1.3, len(x))).astype(np.float32)
    batch = RolloutBatch(
        x=x,
        z=step.z,
        u=step.u,
        old_log_prob=step.log_prob,
        old_value=step.value,
        returns=returns,
        advantages=(returns - step.value).astype(np.float32),
        gate_stage=stages,
        actor_update_mask=step.actor_update_mask,
        run_id="A1-SYNTHETIC-ONLY",
        profile=settings.profile,
    )
    metrics = engine.update(batch)
    assert metrics["actor_optimizer_steps"] > 0
    assert metrics["critic_optimizer_steps"] > 0
    updated = engine.act(x, stages, deterministic=True)
    np.testing.assert_array_equal(
        updated.u[~step.actor_update_mask], reference[~step.actor_update_mask]
    )
    checkpoint = output / "synthetic_checkpoint.pt"
    torch.save(engine.training_state_dict(), checkpoint)
    restored = StagePPOEngine(settings)
    restored.load_training_state_dict(torch.load(checkpoint, weights_only=True))
    original_sample, restored_sample = engine.act(x, stages), restored.act(x, stages)
    for name in ["z", "u", "log_prob", "value"]:
        np.testing.assert_array_equal(
            getattr(original_sample, name), getattr(restored_sample, name)
        )
    artifact = output / "synthetic_stage_actor.npz"
    manifest = engine.export_npz(
        artifact,
        {
            "run_id": "A1-SYNTHETIC-ONLY",
            "policy_commit": "working-tree-A1-not-published",
            "training_commit": "synthetic-harness-not-team-B-racing-runner",
            "training_kind": "synthetic-smoke",
        },
    )
    policy = NumpyStagePolicy(artifact, manifest["artifact_sha256"])
    numpy_u = policy.commands(x, stages)
    export_error = float(np.max(np.abs(numpy_u - updated.u)))
    np.testing.assert_allclose(numpy_u, updated.u, rtol=1e-5, atol=1e-5)
    assert file_sha256(ORIGINAL_MODEL) == ORIGINAL_SHA256
    report = {
        "status": "A1_SYNTHETIC_SMOKE_PASSED",
        "racing_training_performed": False,
        "race_success_rate": None,
        "race_lap_time": None,
        "warning": "Synthetic targets only. Do not use these weights as a racing candidate.",
        "output_directory": str(output.resolve()),
        "metrics": metrics,
        "checkpoint_sampling_restored": True,
        "inactive_stages_unchanged": True,
        "export_max_abs_error": export_error,
        "original_model_unchanged": True,
        "artifact_manifest": manifest,
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    """Run once into a unique temporary directory unless a new path is requested."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or (
        Path(tempfile.gettempdir()) / "drone-team-a-smoke" / uuid.uuid4().hex[:12]
    )
    print(json.dumps(run_smoke(output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
