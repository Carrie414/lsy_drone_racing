"""Check the team entry against the original actor on shared real simulation observations.

This is a deployment compatibility smoke check, not training or an independent
performance evaluation. Both actors see the same observations; only the team
entry's action advances the environment. No reward or official rule is changed.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
import uuid
from pathlib import Path

import gymnasium
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.control.level2_reference import Level2Reference
from lsy_drone_racing.control.team_policy_controller import TeamPolicyController
from lsy_drone_racing.utils import load_config


def run_check(seeds: list[int], output: Path) -> dict:
    """Run complete episodes from the normal start and compare every issued action."""
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "config/level2.toml")
    config.sim.render = False  # Same rendering-only override as official sim.py.
    rows = []
    for seed in seeds:
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
                seed=seed,
            )
        )
        max_error, max_control_ms = 0.0, 0.0
        try:
            obs, info = env.reset()
            entry = TeamPolicyController(obs, info, config)
            baseline = Level2Reference(obs, info, config)
            steps = 0
            while True:
                started = time.perf_counter()
                action = entry.compute_control(obs, info)
                control_ms = (time.perf_counter() - started) * 1000
                reference = baseline.compute_control(obs, info)
                max_error = max(max_error, float(np.max(np.abs(action - reference))))
                np.testing.assert_allclose(action, reference, atol=1e-5, rtol=1e-5)
                obs, reward, terminated, truncated, info = env.step(action)
                started = time.perf_counter()
                callback_finished = entry.step_callback(
                    action, obs, reward, terminated, truncated, info
                )
                callback_ms = (time.perf_counter() - started) * 1000
                assert callback_finished is False
                max_control_ms = max(max_control_ms, control_ms + callback_ms)
                steps += 1
                if terminated or truncated:
                    break
                if steps > 1500:
                    raise AssertionError("Official Level 2 episode did not terminate on time")
            row = {
                "seed": seed,
                "steps": steps,
                "gate_passes": int(obs["n_gates_passed"]),
                "success": bool(obs["n_gates_passed"] == len(obs["gate_sequence"])),
                "official_time_s": (steps - 1) / config.env.freq,
                "max_action_abs_error": max_error,
                "max_entry_control_and_callback_ms": max_control_ms,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
        finally:
            env.close()
    report = {
        "status": "REAL_OBSERVATION_DEPLOYMENT_COMPATIBILITY_PASSED",
        "training_performed": False,
        "purpose": "shared-observation action equivalence, not independent race comparison",
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report


def main() -> None:
    """Run on a small diagnostic set without changing any official config file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[75131, 75132, 75133])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or (
        Path(tempfile.gettempdir()) / "drone-team-a-deployment" / f"{uuid.uuid4().hex}.json"
    )
    report = run_check(args.seeds, output)
    print(report["status"])
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
