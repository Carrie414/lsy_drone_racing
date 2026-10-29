"""A's specialized stage mask, frozen fallback and serialized deployment contract."""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING

import numpy as np
import torch

from training.team_a.engine import StagePPOEngine
from training.team_a.stage_export import NumpyStagePolicy
from training.team_a.train_first_gate import first_gate_settings
from training.team_b.train_racing import run_training

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_first_gate_zero_and_nonzero_residual_export(tmp_path: Path) -> None:
    """A learned correction never changes inactive stages at the same public input."""
    engine = StagePPOEngine(first_gate_settings(20261029))
    x = np.tile(engine.reference.obs_mean.numpy(), (6, 1)).astype(np.float32)
    x[:, 42:] = 0
    stage = np.arange(6, dtype=np.int64)
    base = engine.reference.deterministic(x)
    initial = engine.act(x, stage, deterministic=True)
    np.testing.assert_allclose(initial.u, base, atol=1e-6)
    np.testing.assert_array_equal(
        initial.actor_update_mask, [True, False, False, False, False, False]
    )
    with torch.no_grad():
        engine.actor.network[-1].bias.fill_(0.5)
    corrected = engine.act(x, stage, deterministic=True)
    np.testing.assert_array_equal(corrected.u[1:], base[1:])
    assert np.all(np.abs(corrected.u[0] - base[0]) <= np.array([0.08, 0.08, 0.05, 0.08]) + 1e-6)
    path = tmp_path / "first-gate.npz"
    manifest = engine.export_npz(
        path,
        {
            "run_id": "first-gate-test",
            "policy_commit": "test",
            "training_commit": "test",
            "training_kind": "synthetic-smoke",
        },
    )
    runtime = NumpyStagePolicy(path, manifest["artifact_sha256"])
    assert runtime.active_stages.tolist() == [0]
    np.testing.assert_allclose(runtime.commands(x, stage), corrected.u, atol=1e-6)


def test_custom_settings_survive_checkpoint_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep the shared runner's profile override on export and checkpoint resume."""
    from tests.team_b.test_integration import FakeRace
    from training.team_b import train_racing

    monkeypatch.setattr(train_racing, "make_environment", lambda *_: FakeRace())
    settings = first_gate_settings(20261029)
    args = argparse.Namespace(
        candidate="A4-first-gate",
        updates=1,
        batch_steps=6,
        seed_pool=4,
        seed=settings.seed,
        output_dir=tmp_path / "initial",
        resume=None,
    )
    run_training(args, settings=settings)
    first = json.loads((args.output_dir / "candidate.manifest.json").read_text())
    assert first["active_stages"] == [0]
    args.resume = args.output_dir / "checkpoint.pt"
    args.output_dir = tmp_path / "continued"
    args.updates = 2
    result = run_training(args, settings=settings)
    assert result["total_steps"] == 12
    assert result["total_updates"] == 2
