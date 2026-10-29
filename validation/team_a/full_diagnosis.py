"""Complete public-observation diagnostics of C0/C3/A4; no policy changes."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.level2_reference import reference_features
from lsy_drone_racing.control.team_policy_runtime import NumpyStagePolicy
from lsy_drone_racing.envs.race_core import build_action_space
from lsy_drone_racing.utils import load_config
from training.team_b.reward import racing_reward
from validation.team_a.diagnose_race import describe, durations

ROOT = Path(__file__).resolve().parents[2]


def rows(path: Path) -> list[dict]:
    """Read an existing local JSONL without opening a simulator."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def angle(a: np.ndarray, b: np.ndarray) -> float | None:
    """Angle between nonzero vectors, avoiding undefined headings at zero speed."""
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(a @ b / norm, -1, 1)))) if norm > 1e-8 else None


def crossing(step: dict) -> dict | None:
    """Interpolate a passed plane using only publicly revealed geometry.

    A terminal warp is never interpreted as physical motion. The interpolation
    is observational and does not replace the official gate counter.
    """
    obs, after = step["obs"], step["next_obs"]
    stage = int(step["gate_stage"])
    if stage >= len(obs["gate_sequence"]) or step["terminated"] or step["truncated"]:
        return None
    if int(after["n_gates_passed"]) != stage + 1:
        return None
    gate = int(obs["gate_sequence"][stage])
    if not after["gates_visited"][gate]:
        return None
    frame = Rotation.from_quat(after["gates_quat"][gate]).as_matrix()
    center = np.asarray(after["gates_pos"][gate])
    a = frame.T @ (np.asarray(obs["pos"]) - center)
    b = frame.T @ (np.asarray(after["pos"]) - center)
    if abs(b[0] - a[0]) < 1e-8:
        return None
    fraction = -a[0] / (b[0] - a[0])
    if not 0 <= fraction <= 1:
        return None
    local = a + fraction * (b - a)
    position = center + frame @ local
    velocity = np.asarray(obs["vel"]) + fraction * (np.asarray(after["vel"]) - obs["vel"])
    result = {
        "stage": stage,
        "step": step["step"],
        "event_time_s": (step["step"] + fraction) / 50,
        "gate_center": center.tolist(),
        "gate_quat": after["gates_quat"][gate],
        "position": position.tolist(),
        "velocity": velocity.tolist(),
        "speed_mps": float(np.linalg.norm(velocity)),
        "vertical_speed_mps": float(velocity[2]),
        "local_lateral_m": float(local[1]),
        "local_vertical_m": float(local[2]),
        "point_margin_m": float(0.2 - max(abs(local[1]), abs(local[2]))),
        "normal_speed_mps": float(velocity @ frame[:, 0] * obs["gate_sequence_direction"][stage]),
    }
    if stage + 1 < len(obs["gate_sequence"]):
        nxt = int(obs["gate_sequence"][stage + 1])
        next_normal = Rotation.from_quat(after["gates_quat"][nxt]).as_matrix()[:, 0]
        next_normal *= obs["gate_sequence_direction"][stage + 1]
        result.update(
            next_gate_known=bool(after["gates_visited"][nxt]),
            heading_to_next_center_deg=angle(
                velocity, np.asarray(after["gates_pos"][nxt]) - position
            ),
            heading_to_next_entry_normal_deg=angle(velocity, next_normal),
        )
    return result


def bootstrap_mean(a: np.ndarray) -> dict:
    """Resample whole paired episodes; descriptive CI, not an unbiased selection test."""
    a = np.asarray(a, dtype=float)
    if not len(a):
        return {"n": 0}
    rng = np.random.default_rng(20261030)
    means = a[rng.integers(0, len(a), size=(2000, len(a)))].mean(axis=1)
    return {
        "n": len(a),
        "mean": float(a.mean()),
        "bootstrap95": np.percentile(means, [2.5, 97.5]).tolist(),
    }


def matched(base: dict, other: dict) -> dict:
    """Keep failures separate from timing of common successful scenes."""
    common = [s for s in base if base[s]["success"] and other[s]["success"]]
    recovered = [s for s in base if not base[s]["success"] and other[s]["success"]]
    lost = [s for s in base if base[s]["success"] and not other[s]["success"]]
    changes = np.asarray([durations(other[s]) - durations(base[s]) for s in common]).reshape(-1, 6)
    n = len(recovered) + len(lost)
    p = (
        min(1.0, 2 * sum(comb(n, k) for k in range(min(len(recovered), len(lost)) + 1)) / 2**n)
        if n
        else 1.0
    )
    return {
        "recovered": recovered,
        "lost": lost,
        "common_successes": common,
        "paired_success_exact_two_sided_p": p,
        "time_delta_s": bootstrap_mean(changes.sum(axis=1)),
        "stage_time_delta_s": [bootstrap_mean(changes[:, stage]) for stage in range(6)],
        "first_two_stages_delta_s": bootstrap_mean(changes[:, :2].sum(axis=1)),
    }


def reward_order_audit(episodes: list[dict]) -> dict:
    """Check same-seed successful pairs: does a slower flight earn more reward?"""
    groups = defaultdict(list)
    for episode in episodes:
        if episode["success"]:
            groups[episode["seed"]].append(episode)
    inversions, total = [], 0
    for seed, group in groups.items():
        for a, b in combinations(group, 2):
            fast, slow = sorted([a, b], key=lambda row: row["official_time"])
            if slow["official_time"] - fast["official_time"] < 0.04 - 1e-8:
                continue
            total += 1
            if slow["training_reward"] > fast["training_reward"] + 1e-8:
                inversions.append(
                    {
                        "environment_seed": seed,
                        "fast": {
                            key: fast[key]
                            for key in (
                                "update",
                                "official_time",
                                "training_reward",
                                "reward_components",
                            )
                        },
                        "slow": {
                            key: slow[key]
                            for key in (
                                "update",
                                "official_time",
                                "training_reward",
                                "reward_components",
                            )
                        },
                    }
                )
    return {
        "same_seed_successful_pairs_at_least_40ms_apart": total,
        "slower_higher_reward_pairs": len(inversions),
        "fraction": len(inversions) / total if total else None,
        "examples": inversions[:10],
    }


def main() -> None:
    """Analyze existing C0 and freshly replayed C3/A4 development trajectories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_root
    config = load_config(ROOT / "config/level2.toml")
    space = build_action_space("attitude", config.sim.drone, config.sim.dynamics)
    low, high = float(space.low[3]), float(space.high[3])
    folders = {
        "C0": ROOT.parents[1] / "outputs/team_training_20261029/A-first-gate-C0-traces",
        "C3": out / "C3-traces",
        "A4": out / "A4-traces",
    }
    # ROOT is .../work/carrie-level2-fast-upload; parents[1] is the task workspace.
    records = {
        name: {row["eval_seed"]: row for row in rows(folder / "episodes.jsonl")}
        for name, folder in folders.items()
    }
    assert all(set(record) == set(records["C0"]) for record in records.values())
    events, failures, response, reconstructed = {}, [], {}, {}
    segment_rows, reveal_rows = [], []
    reference_path = ROOT / "lsy_drone_racing/control/policies_team/team_policy_baseline.npz"
    reference = NumpyStagePolicy(
        reference_path, "e965ad393e1f96442653505d0c507705cfd52960ea4b250a27694d61b40abf05"
    )
    a4 = NumpyStagePolicy(
        ROOT / "lsy_drone_racing/control/policies_team/team_a4_first_gate.npz",
        "13678eca8dc97c910307dfbcf27ad1998adbfd0045af0a1ae891d45559245618",
    )
    direct_deltas, direct_stages = [], []
    for name, folder in folders.items():
        event_map = {}
        reconstructed[name] = {}
        lag_errors = {stage: [[] for _ in range(11)] for stage in range(5)}
        for seed, ep in records[name].items():
            with gzip.open(folder / ep["trace_path"], "rt", encoding="utf-8") as stream:
                trace = [json.loads(line) for line in stream]
            assert len(trace) == ep["control_steps"]
            stages = np.array([int(step["gate_stage"]) for step in trace])
            commands = np.asarray([step["action"] for step in trace])
            actual_up = Rotation.from_quat(
                np.asarray([step["obs"]["quat"] for step in trace])
            ).as_matrix()[:, :, 2]
            desired_up = Rotation.from_euler("xyz", commands[:, :3]).as_matrix()[:, :, 2]
            normalized = commands.astype(np.float64).copy()
            normalized[:, :2] /= 1.5
            normalized[:, 2] /= 0.5
            normalized[:, 3] = 2 * (normalized[:, 3] - low) / (high - low) - 1
            normalized = np.clip(normalized, -1, 1).astype(np.float32)
            reward_values, reward_parts = [], defaultdict(float)
            for index, step in enumerate(trace):
                value, parts = racing_reward(
                    step["obs"],
                    step["next_obs"],
                    normalized[index],
                    normalized[index - 1] if index else np.zeros(4, dtype=np.float32),
                    terminated=step["terminated"],
                    truncated=step["truncated"],
                )
                reward_values.append(value)
                for key, part in parts.items():
                    reward_parts[key] += part
            reconstructed[name][seed] = {
                "raw_sum": float(sum(reward_values)),
                "discounted_G0_gamma_099": float(
                    np.asarray(reward_values) @ (0.99 ** np.arange(len(trace)))
                ),
                "components": dict(reward_parts),
            }
            for stage in range(5):
                idx = np.flatnonzero(stages == stage)
                if not len(idx):
                    continue
                assert abs(len(idx) / 50 - durations(ep)[stage]) < 1e-10
                control = commands[idx]
                velocities = np.array([trace[i]["obs"]["vel"] for i in idx])
                speed = np.linalg.norm(velocities, axis=1)
                valid = [
                    trace[i]
                    for i in idx
                    if not trace[i]["terminated"] and not trace[i]["truncated"]
                ]
                distance = sum(
                    np.linalg.norm(np.asarray(step["next_obs"]["pos"]) - step["obs"]["pos"])
                    for step in valid
                )
                u = 2 * (control[:, 3] - low) / (high - low) - 1
                segment_rows.append(
                    {
                        "policy": name,
                        "seed": seed,
                        "stage": stage,
                        "success": ep["success"],
                        "seconds": len(idx) / 50,
                        "distance_m": float(distance),
                        "mean_speed_mps": float(speed.mean()),
                        "min_speed_mps": float(speed.min()),
                        "thrust_upper_fraction": float(np.mean(u >= 0.99)),
                        "thrust_lower_fraction": float(np.mean(u <= -0.99)),
                        "tilt_bound_fraction": float(
                            np.mean(np.any(np.abs(control[:, :2]) >= 1.485, axis=1))
                        ),
                        "max_control_ms": max(trace[i]["total_control_ms"] for i in idx),
                    }
                )
                # Fixed windows for all lags prevent different samples at each candidate shift.
                starts = idx[idx + 10 <= idx[-1]]
                for lag in range(11):
                    if len(starts):
                        dots = np.sum(desired_up[starts] * actual_up[starts + lag], axis=1)
                        lag_errors[stage][lag].append(
                            float(np.degrees(np.arccos(np.clip(dots, -1, 1))).mean())
                        )
                gate = int(trace[int(idx[0])]["obs"]["gate_sequence"][stage])
                known = [int(i) for i in idx if trace[int(i)]["obs"]["gates_visited"][gate]]
                update = next(
                    (
                        trace[int(i)]
                        for i in idx
                        if not trace[int(i)]["obs"]["gates_visited"][gate]
                        and trace[int(i)]["next_obs"]["gates_visited"][gate]
                    ),
                    None,
                )
                reveal_rows.append(
                    {
                        "policy": name,
                        "seed": seed,
                        "stage": stage,
                        "success": ep["success"],
                        "actionable_steps": int(idx[-1]) - known[0] + 1 if known else 0,
                        "target_known_at_last_action": bool(known),
                        "update_jump_m": float(
                            np.linalg.norm(
                                np.asarray(update["next_obs"]["gates_pos"][gate])
                                - update["obs"]["gates_pos"][gate]
                            )
                        )
                        if update
                        else 0.0,
                    }
                )
            event_map[seed] = {}
            for i, step in enumerate(trace):
                event = crossing(step)
                if event is not None:
                    event["policy"], event["seed"] = name, seed
                    event_map[seed][event["stage"]] = event
                if name == "C0" and i % 5 == 0:
                    obs = {key: np.asarray(value) for key, value in step["obs"].items()}
                    previous = np.zeros(4, dtype=np.float32)
                    if i:
                        previous = np.asarray(trace[i - 1]["action"], dtype=np.float64)
                        previous[:2] /= 1.5
                        previous[2] /= 0.5
                        previous[3] = 2 * (previous[3] - low) / (high - low) - 1
                        previous = np.clip(previous, -1, 1).astype(np.float32)
                    x = reference_features(obs, previous)[None]
                    s = np.array([step["gate_stage"]], dtype=np.int64)
                    direct_deltas.append((a4.commands(x, s) - reference.commands(x, s))[0])
                    direct_stages.append(int(s[0]))
            if not ep["success"]:
                last = trace[-1]
                stage = int(last["gate_stage"])
                obs = last["obs"]
                gate = int(obs["gate_sequence"][stage])
                frame = Rotation.from_quat(obs["gates_quat"][gate]).as_matrix()
                local = frame.T @ (np.asarray(obs["pos"]) - obs["gates_pos"][gate])
                failures.append(
                    {
                        "policy": name,
                        "seed": seed,
                        "stage": stage,
                        "time_s": ep["official_time"],
                        "target_known": bool(obs["gates_visited"][gate]),
                        "preterminal_pos": obs["pos"],
                        "preterminal_velocity": obs["vel"],
                        "speed_mps": float(np.linalg.norm(obs["vel"])),
                        "target_local_position": local.tolist(),
                        "action": last["action"],
                        "terminated": last["terminated"],
                        "truncated": last["truncated"],
                        "near_observed_target_plane_15cm": bool(abs(local[0]) <= 0.15)
                        if obs["gates_visited"][gate]
                        else None,
                        "contact_object": "unavailable from public observations and empty info",
                    }
                )
        events[name] = event_map
        response[name] = {}
        for stage in range(5):
            curve = [float(np.mean(a)) if a else None for a in lag_errors[stage]]
            response[name][str(stage)] = {
                "lags_ms": list(range(0, 201, 20)),
                "mean_angle_degrees": curve,
                "episodes": len(lag_errors[stage][0]),
                "empirical_best_alignment_ms": int(np.nanargmin(curve)) * 20,
                "interpretation": (
                    "Observational alignment only, not identified physical delay "
                    "or proof compensation helps"
                ),
            }
    paired = {
        "A4_vs_C0": matched(records["C0"], records["A4"]),
        "A4_vs_C3": matched(records["C3"], records["A4"]),
    }
    exit_comparisons = {}
    for base_name in ("C0", "C3"):
        common = paired[f"A4_vs_{base_name}"]["common_successes"]
        exit_comparisons[base_name] = {}
        for stage in range(5):
            eligible = [
                s for s in common if stage in events[base_name][s] and stage in events["A4"][s]
            ]
            comparison = {}
            for key in (
                "speed_mps",
                "vertical_speed_mps",
                "local_lateral_m",
                "local_vertical_m",
                "point_margin_m",
                "heading_to_next_center_deg",
                "heading_to_next_entry_normal_deg",
            ):
                pairs = [
                    (events[base_name][s][stage].get(key), events["A4"][s][stage].get(key))
                    for s in eligible
                ]
                comparison[key] = bootstrap_mean(
                    np.array([b - a for a, b in pairs if a is not None and b is not None])
                )
            comparison["next_gate_known_in_both"] = sum(
                events[base_name][s][stage].get("next_gate_known", False)
                and events["A4"][s][stage].get("next_gate_known", False)
                for s in eligible
            )
            for key in ("heading_to_next_center_deg", "heading_to_next_entry_normal_deg"):
                differences = [
                    events["A4"][s][stage][key] - events[base_name][s][stage][key]
                    for s in eligible
                    if events["A4"][s][stage].get("next_gate_known", False)
                    and events[base_name][s][stage].get("next_gate_known", False)
                ]
                comparison["known_next_" + key] = bootstrap_mean(np.asarray(differences))
            exit_comparisons[base_name][str(stage)] = comparison
    frozen_reward_order = {}
    for base_name in ("C0", "C3"):
        common = paired[f"A4_vs_{base_name}"]["common_successes"]
        cases = []
        for seed in common:
            first, second = records[base_name][seed], records["A4"][seed]
            if abs(first["official_time"] - second["official_time"]) < 0.02 - 1e-8:
                continue
            fast, slow = (
                (base_name, "A4")
                if first["official_time"] < second["official_time"]
                else ("A4", base_name)
            )
            cases.append(
                {
                    "seed": seed,
                    "faster_policy": fast,
                    "slower_policy": slow,
                    "raw_slower_higher": reconstructed[slow][seed]["raw_sum"]
                    > reconstructed[fast][seed]["raw_sum"],
                    "discounted_slower_higher": reconstructed[slow][seed]["discounted_G0_gamma_099"]
                    > reconstructed[fast][seed]["discounted_G0_gamma_099"],
                    "fast_return": reconstructed[fast][seed],
                    "slow_return": reconstructed[slow][seed],
                    "fast_time": records[fast][seed]["official_time"],
                    "slow_time": records[slow][seed]["official_time"],
                }
            )
        frozen_reward_order[base_name] = {
            "different_time_pairs_at_least_20ms": len(cases),
            "raw_sum_slower_higher": sum(row["raw_slower_higher"] for row in cases),
            "discounted_G0_slower_higher": sum(row["discounted_slower_higher"] for row in cases),
            "cases": cases,
        }
    training_root = ROOT.parents[1] / "outputs/a_first_gate_training_20261029"
    reward_checks, training_checks = {}, {}
    for seed in (20261029, 20261030):
        training = sum(
            [rows(training_root / f"A4-{seed}-u{u}/episodes.jsonl") for u in (32, 64)], []
        )
        updates = sum([rows(training_root / f"A4-{seed}-u{u}/updates.jsonl") for u in (32, 64)], [])
        reward_checks[str(seed)] = reward_order_audit(training)
        training_checks[str(seed)] = {
            "updates": len(updates),
            "episodes": len(training),
            "actor_samples": sum(row["actor_samples"] for row in updates),
            "total_samples": sum(row["samples"] for row in updates),
            "kl_early_stops": sum(row["early_stopped_kl"] for row in updates),
            "mean_anchor_mse": float(np.mean([row["anchor_mse"] for row in updates])),
            "exploratory_failures_by_stage": {
                str(s): sum(not row["success"] and row["gate_pass_count"] == s for row in training)
                for s in range(5)
            },
        }
    stage_array, deltas = np.asarray(direct_stages), np.asarray(direct_deltas)
    result = {
        "scope": "Three frozen policies, 200 development seeds each; C0 reused, C3/A4 replayed",
        "paired": paired,
        "exit_state_comparisons": exit_comparisons,
        "response_alignment": response,
        "reward_order_audit": reward_checks,
        "frozen_replay_reward_order": frozen_reward_order,
        "training_audit": training_checks,
        "direct_same_input_action_delta": {
            str(s): {
                "samples": int(np.sum(stage_array == s)),
                "mean_abs_per_axis": np.abs(deltas[stage_array == s]).mean(axis=0).tolist(),
                "max_abs_per_axis": np.abs(deltas[stage_array == s]).max(axis=0).tolist(),
            }
            for s in range(6)
        },
        "summaries": {
            name: json.loads((folder / "summary.json").read_text())
            for name, folder in folders.items()
        },
        "reveal_summary": {
            name: {
                str(stage): describe(
                    [
                        row["actionable_steps"]
                        for row in reveal_rows
                        if row["policy"] == name and row["stage"] == stage and not row["success"]
                    ]
                )
                for stage in range(5)
            }
            for name in folders
        },
        "validation_opened": False,
        "final_opened": False,
        "training_performed": False,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    for filename, value in [
        ("complete_diagnosis.json", result),
        ("segments.json", segment_rows),
        ("gate_events.json", events),
        ("failures.json", failures),
        ("revelations.json", reveal_rows),
        ("reconstructed_training_reward.json", reconstructed),
    ]:
        (out / filename).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: result[key] for key in ("paired", "reward_order_audit", "training_audit")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
