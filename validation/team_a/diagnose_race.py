"""Describe race bottlenecks from existing development observations; no training."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from lsy_drone_racing.envs.race_core import build_action_space
from lsy_drone_racing.utils import load_config

ROOT = Path(__file__).resolve().parents[2]


def durations(episode: dict) -> np.ndarray:
    """Count control intervals in each stage, including post-completion stage 5."""
    counts = np.zeros(6, dtype=np.float64)
    current, start = 0, 0
    for change in episode["transition_commands"]:
        counts[current] += change["step"] - start
        current, start = change["to_stage"], change["step"]
    counts[current] += episode["control_steps"] - start
    return counts / 50


def describe(values: list | np.ndarray) -> dict:
    """Return descriptive quantiles without implying statistical independence."""
    a = np.asarray(values, dtype=float)
    return {
        "n": len(a),
        "mean": float(a.mean()) if len(a) else None,
        "p10": float(np.percentile(a, 10)) if len(a) else None,
        "median": float(np.median(a)) if len(a) else None,
        "p90": float(np.percentile(a, 90)) if len(a) else None,
    }


def main() -> None:
    """Write segment metrics, paired stage times and explicit interpretation limits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Use a new output directory")
    args.output_dir.mkdir(parents=True)
    config = load_config(ROOT / "config/level2.toml")
    space = build_action_space("attitude", config.sim.drone, config.sim.dynamics)
    low, high = float(space.low[3]), float(space.high[3])
    episodes = [
        json.loads(line) for line in (args.trace_dir / "episodes.jsonl").read_text().splitlines()
    ]
    candidate = {
        row["eval_seed"]: row
        for row in map(json.loads, (args.candidate_dir / "episodes.jsonl").read_text().splitlines())
    }
    if {row["eval_seed"] for row in episodes} != set(candidate):
        raise ValueError("Exact same development seeds required")
    rows, gate_rows = [], []
    for episode in episodes:
        with gzip.open(args.trace_dir / episode["trace_path"], "rt", encoding="utf-8") as stream:
            trace = [json.loads(line) for line in stream]
        for stage in range(5):
            portion = [step for step in trace if step["gate_stage"] == stage]
            if not portion:
                continue
            speed = np.array([np.linalg.norm(step["obs"]["vel"]) for step in portion])
            commands = np.array([step["action"] for step in portion])
            thrust_u = 2 * (commands[:, 3] - low) / (high - low) - 1
            valid_steps = [
                step for step in portion if not step["terminated"] and not step["truncated"]
            ]
            travel = sum(
                np.linalg.norm(np.asarray(step["next_obs"]["pos"]) - step["obs"]["pos"])
                for step in valid_steps
            )
            end_pos = portion[-1]["obs"]["pos"]
            if valid_steps and valid_steps[-1] is portion[-1]:
                end_pos = portion[-1]["next_obs"]["pos"]
            chord = np.linalg.norm(np.asarray(end_pos) - portion[0]["obs"]["pos"])
            velocities = np.array([step["obs"]["vel"] for step in portion])
            turn = 0.0
            for index in range(1, len(velocities)):
                if min(speed[index - 1], speed[index]) >= 1:
                    cosine = np.dot(velocities[index - 1], velocities[index]) / (
                        speed[index - 1] * speed[index]
                    )
                    turn += float(np.arccos(np.clip(cosine, -1, 1)))
            tilt_errors = []
            for step in valid_steps:
                desired_up = Rotation.from_euler("xyz", step["action"][:3]).as_matrix()[:, 2]
                actual_up = Rotation.from_quat(step["next_obs"]["quat"]).as_matrix()[:, 2]
                tilt_errors.append(
                    float(np.degrees(np.arccos(np.clip(desired_up @ actual_up, -1, 1))))
                )
            rows.append(
                {
                    "seed": episode["eval_seed"],
                    "success": episode["success"],
                    "stage": stage,
                    "seconds": len(portion) / 50,
                    "distance_m": float(travel),
                    "chord_m": float(chord),
                    "path_to_chord": float(travel / chord) if chord > 0.05 else None,
                    "mean_speed": float(speed.mean()),
                    "min_speed": float(speed.min()),
                    "p10_speed": float(np.percentile(speed, 10)),
                    "max_speed": float(speed.max()),
                    "speed_below_1_fraction": float(np.mean(speed < 1)),
                    "thrust_upper_fraction": float(np.mean(thrust_u >= 0.99)),
                    "thrust_lower_fraction": float(np.mean(thrust_u <= -0.99)),
                    "tilt_command_bound_fraction": float(
                        np.mean(np.any(np.abs(commands[:, :2]) >= 1.485, axis=1))
                    ),
                    "direction_change_degrees": float(np.degrees(turn)),
                    "mean_next_step_tilt_error_degrees": float(np.mean(tilt_errors))
                    if tilt_errors
                    else None,
                }
            )
        for step in trace:
            stage = int(step["gate_stage"])
            if stage >= 5 or int(step["next_obs"]["n_gates_passed"]) != stage + 1:
                continue
            obs, following = step["obs"], step["next_obs"]
            gate = int(obs["gate_sequence"][stage])
            if not following["gates_visited"][gate] or step["terminated"]:
                continue
            matrix = Rotation.from_quat(following["gates_quat"][gate]).as_matrix()
            a = matrix.T @ (np.asarray(obs["pos"]) - following["gates_pos"][gate])
            b = matrix.T @ (np.asarray(following["pos"]) - following["gates_pos"][gate])
            if abs(b[0] - a[0]) < 1e-8:
                continue
            alpha = -a[0] / (b[0] - a[0])
            if not 0 <= alpha <= 1:
                continue
            cross = a + alpha * (b - a)
            gate_rows.append(
                {
                    "seed": episode["eval_seed"],
                    "success": episode["success"],
                    "stage": stage,
                    "center_lateral_m": float(cross[1]),
                    "center_vertical_m": float(cross[2]),
                    "point_opening_margin_m": float(0.2 - max(abs(cross[1]), abs(cross[2]))),
                }
            )
    summary = {}
    metrics = [
        "seconds",
        "distance_m",
        "path_to_chord",
        "mean_speed",
        "p10_speed",
        "max_speed",
        "speed_below_1_fraction",
        "thrust_upper_fraction",
        "thrust_lower_fraction",
        "tilt_command_bound_fraction",
        "direction_change_degrees",
        "mean_next_step_tilt_error_degrees",
    ]
    for stage in range(5):
        complete = [row for row in rows if row["stage"] == stage and row["success"]]
        crossings = [row for row in gate_rows if row["stage"] == stage and row["success"]]
        summary[str(stage)] = {
            key: describe([row[key] for row in complete if row[key] is not None]) for key in metrics
        }
        summary[str(stage)]["point_opening_margin_m"] = describe(
            [row["point_opening_margin_m"] for row in crossings]
        )
    common = [row for row in episodes if row["success"] and candidate[row["eval_seed"]]["success"]]
    delta = np.array([durations(candidate[row["eval_seed"]]) - durations(row) for row in common])
    result = {
        "baseline_successes": sum(row["success"] for row in episodes),
        "candidate_successes": sum(row["success"] for row in candidate.values()),
        "successful_baseline_stage_metrics": summary,
        "paired_common_successes": len(common),
        "paired_mean_stage_time_change_seconds": delta.mean(axis=0).tolist(),
        "paired_mean_total_time_change_seconds": float(delta.sum(axis=1).mean()),
        "failed_stage_counts": {
            str(stage): sum(
                not row["success"] and row["gate_pass_count"] == stage for row in episodes
            )
            for stage in range(5)
        },
        "limits": [
            "Stage durations count control intervals, unlike official zero-based final-step time; "
            "paired differences agree.",
            "Failed-stage durations are censored and excluded from successful timing comparisons.",
            "Path/chord is descriptive: a straight chord ignores obstacles, gate orientation "
            "and dynamics.",
            "Low speed, turns and saturation are not proof of wasted time or a physical limit.",
            "Gate margin uses observed interpolation for a point center; "
            "no rotor/body clearance guarantee.",
            "Only C0 has full motion traces here; candidate comparison uses paired stage "
            "boundaries, not inferred candidate paths.",
            "Next-step tilt error includes commanded transients and is not an identified latency.",
        ],
    }
    (args.output_dir / "diagnosis.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, data in [("segments.csv", rows), ("crossings.csv", gate_rows)]:
        with (args.output_dir / name).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
