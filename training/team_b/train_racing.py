"""Integration runner: A's stage PPO with B's actual racing reward and GAE."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from lsy_drone_racing.utils import load_config
from training.team_a.engine import PPOSettings, StagePPOEngine
from training.team_a.npz_io import ORIGINAL_MODEL, ORIGINAL_SHA256, file_sha256
from training.team_b.curriculum import HardCaseSampler
from training.team_b.reward import RewardWeights
from training.team_b.rollout import RaceCollector, make_environment

ROOT = Path(__file__).resolve().parents[2]
RUNNER_VERSION = "integrated-racing-v1"


def git_output(*arguments: str) -> str:
    """Query only this exact repository, supporting sandbox ownership differences."""
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", *arguments], cwd=ROOT, text=True
    ).strip()


def write_json(path: Path, value: dict) -> None:
    """Write an experiment record with explicit UTF-8 encoding."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_training(args: argparse.Namespace) -> dict:
    """Train a finite C1/C2/C3 pilot while keeping validation and final sets closed."""
    if args.output_dir.exists():
        raise FileExistsError("Use a new output directory, including when resuming")
    if args.updates < 1 or args.batch_steps < 1 or not 1 <= args.seed_pool <= 4096:
        raise ValueError("Invalid finite training budget or seed pool size")
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    settings = PPOSettings(
        seed=args.seed,
        profile=f"{args.candidate}-stage-ppo-v1",
        epochs=4,
        minibatch_size=128,
        policy_kind="residual" if args.candidate == "C3" else "finetune",
    )
    engine = StagePPOEngine(settings)
    # A finite pool avoids allocating a million Python sampler records. All are training-only seeds.
    allowed = list(range(1000001, 1000001 + args.seed_pool))
    sampler = HardCaseSampler(
        allowed, seed=args.seed, normal_fraction=1.0 if args.candidate == "C1" else 0.5
    )
    initial_update, total_steps, total_episodes = 0, 0, 0
    if args.resume:
        state = torch.load(args.resume, weights_only=True)
        if (
            state["runner_version"] != RUNNER_VERSION
            or state["candidate"] != args.candidate
            or state["batch_steps"] != args.batch_steps
        ):
            raise ValueError("Runner checkpoint/profile/batch size mismatch")
        engine.load_training_state_dict(state["engine"])
        sampler.load_state_dict(state["sampler"])
        random.setstate(state["python_rng"])
        np.random.set_state(
            (
                state["numpy_rng"][0],
                np.asarray(state["numpy_rng"][1], dtype=np.uint32),
                *state["numpy_rng"][2:],
            )
        )
        initial_update, total_steps, total_episodes = (
            state["updates"],
            state["total_steps"],
            state["total_episodes"],
        )
        if args.updates <= initial_update:
            raise ValueError("--updates must exceed the checkpoint's completed update count")
    config = load_config(ROOT / "config/level2.toml")
    config.sim.render = False
    weights = RewardWeights()
    manifest = {
        "runner_version": RUNNER_VERSION,
        "run_id": args.output_dir.name,
        "candidate": args.candidate,
        "source_commit": git_output("rev-parse", "HEAD"),
        "dirty_status": git_output("status", "--short"),
        "settings": asdict(settings),
        "reward_weights": asdict(weights),
        "batch_steps_minimum": args.batch_steps,
        "target_updates": args.updates,
        "training_environment_seeds": allowed,
        "training_initialization_seed": args.seed,
        "sampling": "uniform" if args.candidate == "C1" else "B-dynamic-hardcase-50%-normal",
        "validation_opened": False,
        "final_opened": False,
        "original_model_sha256": ORIGINAL_SHA256,
        "official_config_sha256": file_sha256(ROOT / "config/level2.toml"),
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "environment_resume_semantics": "fresh full-start episode; no serialized environment state",
        "resume_checkpoint": str(args.resume) if args.resume else None,
        "resume_checkpoint_sha256": file_sha256(args.resume) if args.resume else None,
        "source_hashes": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for parent in [ROOT / "training", ROOT / "validation/team_b"]
            for path in sorted(parent.rglob("*.py"))
        },
    }
    write_json(args.output_dir / "manifest.json", manifest)
    env = make_environment(config, allowed[0])
    collector = RaceCollector(env, engine, sampler, args.output_dir.name, weights)
    started = time.perf_counter()
    metrics_rows = []
    try:
        for update in range(initial_update, args.updates):
            collected_at = time.perf_counter()
            batch, episodes = collector.collect(args.batch_steps)
            collection_seconds = time.perf_counter() - collected_at
            metrics = engine.update(batch)
            total_steps += len(batch.x)
            total_episodes += len(episodes)
            row = {
                **metrics,
                "update": update + 1,
                "steps_this_update": len(batch.x),
                "total_steps": total_steps,
                "episodes_this_update": len(episodes),
                "total_episodes": total_episodes,
                "training_exploratory_successes": sum(e["success"] for e in episodes),
                "training_reward_mean": float(np.mean([e["training_reward"] for e in episodes])),
                "collection_seconds": collection_seconds,
                "elapsed_this_run_seconds": time.perf_counter() - started,
            }
            metrics_rows.append(row)
            with (args.output_dir / "updates.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            with (args.output_dir / "episodes.jsonl").open("a", encoding="utf-8") as stream:
                for episode in episodes:
                    stream.write(json.dumps({"update": update + 1, **episode}) + "\n")
            numpy_rng = np.random.get_state()
            checkpoint = {
                "runner_version": RUNNER_VERSION,
                "candidate": args.candidate,
                "batch_steps": args.batch_steps,
                "engine": engine.training_state_dict(),
                "sampler": sampler.state_dict(),
                "updates": update + 1,
                "total_steps": total_steps,
                "total_episodes": total_episodes,
                "python_rng": random.getstate(),
                "numpy_rng": (numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]),
            }
            temporary = args.output_dir / "checkpoint.pending.pt"
            torch.save(checkpoint, temporary)
            os.replace(temporary, args.output_dir / "checkpoint.pt")
            print(json.dumps(row), flush=True)
    finally:
        env.close()
    asset = args.output_dir / "candidate.npz"
    artifact_manifest = engine.export_npz(
        asset,
        {
            "run_id": args.output_dir.name,
            "policy_commit": manifest["source_commit"],
            "training_commit": manifest["source_commit"],
            "training_kind": "racing",
            "candidate": args.candidate,
            "reward_version": "B-v2-minimal",
            "curriculum_version": manifest["sampling"],
            "training_steps": total_steps,
            "training_episodes": total_episodes,
            "initialization_seed": args.seed,
        },
    )
    write_json(args.output_dir / "candidate.manifest.json", artifact_manifest)
    assert file_sha256(ORIGINAL_MODEL) == ORIGINAL_SHA256
    summary = {
        "status": "REAL_RACING_TRAINING_PILOT_COMPLETED",
        "candidate": args.candidate,
        "training_performed": True,
        "validation_opened": False,
        "final_opened": False,
        "total_updates": args.updates,
        "total_steps": total_steps,
        "total_episodes": total_episodes,
        "elapsed_this_run_seconds": time.perf_counter() - started,
        "artifact": str(asset),
        "artifact_sha256": artifact_manifest["artifact_sha256"],
        "original_model_unchanged": True,
        "note": "Exploratory training outcomes are not deterministic race evaluation results.",
        "last_update": metrics_rows[-1],
    }
    write_json(args.output_dir / "summary.json", summary)
    print("SUMMARY " + json.dumps(summary), flush=True)
    return summary


def main() -> None:
    """Run the preregistered finite pilot or continue a trusted local checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=["C1", "C2", "C3"], default="C1")
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--batch-steps", type=int, default=1024)
    parser.add_argument("--seed-pool", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    run_training(parser.parse_args())


if __name__ == "__main__":
    main()
