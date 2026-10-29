"""A-owned first-gate-only profile, reusing B's unchanged rewards and full rollouts."""

from __future__ import annotations

import argparse
from pathlib import Path

from training.team_a.engine import PPOSettings
from training.team_b.train_racing import run_training


def first_gate_settings(seed: int) -> PPOSettings:
    """Freeze stages 1..5 and bound the original actor's first-stage correction."""
    return PPOSettings(
        seed=seed,
        profile="A4-first-gate-residual-v1",
        active_stages=(0,),
        policy_kind="residual",
        residual_bounds=(0.08, 0.08, 0.05, 0.08),
        actor_lr=1e-4,
        initial_std=0.03,
        epochs=4,
        minibatch_size=128,
    )


def main() -> None:
    """Run a finite first-gate pilot with checkpoint-compatible explicit settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=32)
    parser.add_argument("--batch-steps", type=int, default=1024)
    parser.add_argument("--seed-pool", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    args.candidate = "A4-first-gate"
    run_training(args, settings=first_gate_settings(args.seed))


if __name__ == "__main__":
    main()
