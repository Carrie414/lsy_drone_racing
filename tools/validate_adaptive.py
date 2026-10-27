"""Reproduce Level 2 controller checks without modifying the official evaluator."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.utils import load_config, load_controller


def main():
    """Run independently seeded episodes and save timing and completion evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-seed", type=int, default=301)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--controller", default="adaptive_controller.py")
    parser.add_argument("--output", type=Path, default=Path("logs/adaptive-validation.json"))
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "config/level2.toml")
    config.sim.render = False
    controller_class = load_controller(root / "lsy_drone_racing/control" / args.controller)
    env = JaxToNumpy(
        gymnasium.make(
            config.env.id,
            freq=config.env.freq,
            sim_config=config.sim,
            sensor_range=config.env.sensor_range,
            control_mode=config.env.control_mode,
            track=config.env.track,
            disturbances=config.env.get("disturbances"),
            randomizations=config.env.get("randomizations"),
            seed=args.start_seed,
        )
    )
    results = []
    try:
        for seed in range(args.start_seed, args.start_seed + args.runs):
            obs, info = env.reset(seed=seed)
            controller = controller_class(obs, info, config)
            durations = []
            step = 0
            while True:
                started = time.perf_counter()
                action = controller.compute_control(obs, info)
                elapsed = time.perf_counter() - started
                if action.shape != (16,) or not np.all(np.isfinite(action)):
                    raise ValueError("Controller must return 16 finite state-command values")
                obs, reward, terminated, truncated, info = env.step(action)
                started = time.perf_counter()
                finished = controller.step_callback(
                    action, obs, reward, terminated, truncated, info
                )
                durations.append(elapsed + time.perf_counter() - started)
                if terminated or truncated or finished:
                    break
                step += 1
            row = dict(
                seed=seed,
                success=bool(obs["n_gates_passed"] == len(obs["gate_sequence"])),
                gates_passed=int(obs["n_gates_passed"]),
                seconds=step / config.env.freq,
                max_control_ms=max(durations) * 1000,
                over_budget_steps=sum(t > 1 / config.env.freq for t in durations),
            )
            controller.episode_callback()
            controller.episode_reset()
            results.append(row)
            print(json.dumps(row), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    finally:
        env.close()
    successful = [r["seconds"] for r in results if r["success"]]
    print(f"Success: {len(successful)}/{len(results)}")
    if successful:
        print(f"Mean successful lap: {np.mean(successful):.3f}s")
    raise SystemExit(0 if len(successful) / len(results) >= 0.5 else 1)


if __name__ == "__main__":
    main()
