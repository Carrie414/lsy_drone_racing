"""Real A/B module integration on a deterministic fake episode, not a race score."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.envs.race_core import build_action_space
from lsy_drone_racing.utils import load_config
from training.team_a.engine import PPOSettings, StagePPOEngine
from training.team_b.curriculum import HardCaseSampler
from training.team_b.rollout import RaceCollector
from training.team_b.train_racing import run_training

if TYPE_CHECKING:
    import pytest

ROOT = Path(__file__).resolve().parents[2]


class FakeRace:
    """Six transitions include the official post-completion stage 5 control step."""

    def __init__(self) -> None:
        """Use the real public action bounds with a scripted observation stream."""
        config = load_config(ROOT / "config/level2.toml")
        self.action_space = build_action_space("attitude", config.sim.drone, config.sim.dynamics)
        self.step_index = 0
        self.reset_seeds = []

    def observation(self) -> dict:
        """Supply only the public observation structure expected by the original actor."""
        return {
            "pos": np.array([-1.5 + 0.03 * self.step_index, 0.75, 0.1]),
            "quat": np.array([0, 0, 0, 1]),
            "vel": np.zeros(3),
            "ang_vel": np.zeros(3),
            "gates_pos": np.array(
                [[0.5, 0.25, 0.7], [1.05, 0.75, 1.2], [-1, -0.25, 0.7], [0, -0.75, 1.2]]
            ),
            "gates_quat": np.tile([0, 0, 0, 1], (4, 1)),
            "gate_sequence": np.array([0, 1, 2, 3, 1]),
            "gate_sequence_direction": np.ones(5, dtype=np.int64),
            "n_gates_passed": min(5, self.step_index),
            "gates_visited": np.zeros(4, dtype=bool),
            "obstacles_pos": np.empty((0, 3)),
            "obstacles_visited": np.empty(0, dtype=bool),
        }

    def reset(self, *, seed: int) -> tuple[dict, dict]:
        """Start a new full episode and record its training-only seed."""
        self.reset_seeds.append(seed)
        self.step_index = 0
        return self.observation(), {}

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """Advance a scripted test episode without any simulator claims."""
        assert self.action_space.contains(action)
        self.step_index += 1
        return self.observation(), 0.0, self.step_index == 6, False, {}

    def close(self) -> None:
        """Satisfy the runner cleanup interface."""


def test_collector_preserves_terminal_stage_mask_and_episode_boundaries() -> None:
    engine = StagePPOEngine(PPOSettings(epochs=1))
    env = FakeRace()
    collector = RaceCollector(env, engine, HardCaseSampler([1000001, 1000002]), "test")
    batch, episodes = collector.collect(7)
    assert len(episodes) == 2 and len(batch.x) == 12
    np.testing.assert_array_equal(batch.gate_stage, list(range(6)) * 2)
    np.testing.assert_array_equal(
        batch.actor_update_mask, [True, False, True, False, False, False] * 2
    )
    np.testing.assert_allclose(batch.returns[[5, 11]], -0.02, atol=1e-6)
    np.testing.assert_array_equal(batch.x[[0, 6], 42:], 0)
    assert all(e["success"] and e["gate_pass_count"] == 5 for e in episodes)
    metrics = engine.update(batch)
    assert metrics["actor_samples"] == 4 and metrics["actor_optimizer_steps"] > 0


def test_runner_can_resume_total_checkpoint_and_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from training.team_b import train_racing

    monkeypatch.setattr(train_racing, "make_environment", lambda *_: FakeRace())
    first_args = argparse.Namespace(
        candidate="C1",
        updates=1,
        batch_steps=6,
        seed_pool=4,
        seed=20261029,
        output_dir=tmp_path / "first",
        resume=None,
    )
    first = run_training(first_args)
    assert first["total_steps"] == 6
    next_args = argparse.Namespace(
        **{
            **vars(first_args),
            "updates": 2,
            "output_dir": tmp_path / "resumed",
            "resume": first_args.output_dir / "checkpoint.pt",
        }
    )
    resumed = run_training(next_args)
    assert resumed["total_steps"] == 12 and resumed["total_updates"] == 2
    assert (next_args.output_dir / "candidate.manifest.json").exists()
    assert not resumed["validation_opened"] and not resumed["final_opened"]


def test_pilot_screen_rejects_faster_but_less_reliable_candidate(tmp_path: Path) -> None:
    from validation.team_b.compare_pilots import compare

    values = {
        "C0": [(True, 3.4), (True, 3.6), (False, 1.0)],
        "C1": [(True, 3.3), (True, 3.5), (False, 1.0)],
        "C2": [(True, 2.9), (False, 1.0), (False, 1.0)],
    }
    for name, outcomes in values.items():
        folder = tmp_path / f"{name}-windows-development"
        folder.mkdir()
        rows = [
            {"eval_seed": 110001 + i, "success": success, "official_time": elapsed}
            for i, (success, elapsed) in enumerate(outcomes)
        ]
        times = [elapsed for success, elapsed in outcomes if success]
        summary = {
            "runs": 3,
            "successes": len(times),
            "mean_successful_seconds": float(np.mean(times)),
        }
        (folder / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (folder / "episodes.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
        )
    report = compare(tmp_path)
    assert report["worth_further_validation"] == ["C1"]
    assert report["candidates"]["C2"]["baseline_success_candidate_fail"] == 1
    assert not report["validation_opened"] and not report["final_opened"]
