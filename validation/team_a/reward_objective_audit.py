"""Offline reward component and ranking audit; never replace the training reward."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from lsy_drone_racing.envs.race_core import build_action_space
from lsy_drone_racing.utils import load_config
from training.team_b.gae import generalized_advantages
from training.team_b.reward import racing_reward

ROOT = Path(__file__).resolve().parents[2]
GAMMA = 0.99
PARTS = ("progress", "gate", "finish", "time", "failure", "action_change")


def potential(obs: dict, *, terminal: bool = False) -> float:
    """State-only diagnostic potential; terminal/finished states are exactly zero."""
    stage = int(obs["n_gates_passed"])
    if terminal or stage >= len(obs["gate_sequence"]):
        return 0.0
    gate = int(obs["gate_sequence"][stage])
    return float(stage - np.linalg.norm(np.asarray(obs["pos"]) - obs["gates_pos"][gate]))


def potential_terms(trace: list[dict], gamma: float = GAMMA) -> np.ndarray:
    """Correct gamma*Phi(next)-Phi(now), including terminal warps and task deadlines."""
    return np.array(
        [
            gamma * potential(step["next_obs"], terminal=step["terminated"] or step["truncated"])
            - potential(step["obs"])
            for step in trace
        ]
    )


def progress_decomposition(trace: list[dict], actual: float) -> dict | None:
    """Telescope successful progress, splitting target switches and revelation updates."""
    active = [step for step in trace if step["gate_stage"] < len(step["obs"]["gate_sequence"])]
    if any(step["terminated"] or step["truncated"] for step in active):
        return None  # Never use a warped failure position as a physical endpoint.
    goals = [
        np.asarray(step["obs"]["gates_pos"][step["obs"]["gate_sequence"][step["gate_stage"]]])
        for step in active
    ]
    start = float(np.linalg.norm(np.asarray(active[0]["obs"]["pos"]) - goals[0]))
    end = float(np.linalg.norm(np.asarray(active[-1]["next_obs"]["pos"]) - goals[-1]))
    switch, reveal, clipping = 0.0, 0.0, 0.0
    for index, step in enumerate(active):
        p, following = np.asarray(step["obs"]["pos"]), np.asarray(step["next_obs"]["pos"])
        raw = float(np.linalg.norm(p - goals[index]) - np.linalg.norm(following - goals[index]))
        clipping += float(np.clip(raw, -0.5, 0.5) - raw)
        if index:
            change = float(np.linalg.norm(p - goals[index]) - np.linalg.norm(p - goals[index - 1]))
            if active[index - 1]["gate_stage"] != step["gate_stage"]:
                switch += change
            else:
                reveal += change
    reconstructed = start - end + switch + reveal + clipping
    if not np.isclose(actual, reconstructed, atol=1e-8):
        raise AssertionError("Progress decomposition does not match the actual reward")
    return {
        "initial_distance": start,
        "last_target_end_distance": end,
        "target_switch_correction": switch,
        "same_target_observation_update_correction": reveal,
        "clipping_correction": clipping,
        "reconstructed_sum": reconstructed,
    }


def variant_scores(parts: np.ndarray, shaping: np.ndarray) -> dict[str, float]:
    """Score frozen trajectories under diagnostic alternatives, without training them."""
    weights = GAMMA ** np.arange(len(parts))
    discounted = weights @ parts
    base = parts[:, 2] + parts[:, 3] + parts[:, 4]
    return {
        "current": float(discounted.sum()),
        "no_progress": float(discounted.sum() - discounted[0]),
        "no_intermediate_gate_bonus": float(discounted.sum() - discounted[1]),
        "gamma_1_current": float(parts.sum()),
        "time_x5": float(discounted.sum() + 4 * discounted[3]),
        "time_x10": float(discounted.sum() + 9 * discounted[3]),
        "time_x100": float(discounted.sum() + 99 * discounted[3]),
        "completion_time_failure_only": float(weights @ base),
        "completion_time_failure_with_potential": float(weights @ (base + shaping)),
    }


def compare_scores(base: dict, other: dict) -> dict:
    """Count ordering reversals per fixed same-seed pair, not across different courses."""
    names = list(next(iter(base.values()))["variants"])
    result = {
        name: {
            "success_time_pairs": 0,
            "slower_scores_higher": [],
            "success_failure_pairs": 0,
            "failure_scores_higher": [],
        }
        for name in names
    }
    for seed in base:
        a, b = base[seed], other[seed]
        if (
            a["success"]
            and b["success"]
            and abs(a["official_time"] - b["official_time"]) >= 0.02 - 1e-8
        ):
            fast, slow = (a, b) if a["official_time"] < b["official_time"] else (b, a)
            for name in names:
                result[name]["success_time_pairs"] += 1
                if slow["variants"][name] > fast["variants"][name] + 1e-10:
                    result[name]["slower_scores_higher"].append(seed)
        if bool(a["success"]) != bool(b["success"]):
            success, failure = (a, b) if a["success"] else (b, a)
            for name in names:
                result[name]["success_failure_pairs"] += 1
                if failure["variants"][name] > success["variants"][name] + 1e-10:
                    result[name]["failure_scores_higher"].append(seed)
    return result


def main() -> None:
    """Reconstruct all component returns and examine diagnostic ablations on 600 traces."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    assert inspect.signature(generalized_advantages).parameters["gamma"].default == GAMMA
    folders = {
        "C0": ROOT.parents[1] / "outputs/team_training_20261029/A-first-gate-C0-traces",
        "C3": ROOT.parents[1] / "outputs/full_diagnosis_20261030/C3-traces",
        "A4": ROOT.parents[1] / "outputs/full_diagnosis_20261030/A4-traces",
    }
    config = load_config(ROOT / "config/level2.toml")
    space = build_action_space("attitude", config.sim.drone, config.sim.dynamics)
    low, high = float(space.low[3]), float(space.high[3])
    all_records, max_shape_error, inputs = {}, 0.0, {}
    for policy, folder in folders.items():
        episodes = [
            json.loads(line) for line in (folder / "episodes.jsonl").read_text().splitlines()
        ]
        records = {}
        for ep in episodes:
            path = folder / ep["trace_path"]
            inputs[policy + "/" + path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                trace = [json.loads(line) for line in stream]
            previous, components, gate_events = np.zeros(4), [], []
            for step in trace:
                u = np.array(step["action"], dtype=float)
                u[:2] /= 1.5
                u[2] /= 0.5
                u[3] = 2 * (u[3] - low) / (high - low) - 1
                u = np.clip(u, -1, 1)
                reward, parts = racing_reward(
                    step["obs"],
                    step["next_obs"],
                    u,
                    previous,
                    terminated=step["terminated"],
                    truncated=step["truncated"],
                )
                assert abs(sum(parts.values()) - reward) < 1e-10
                components.append([parts[key] for key in PARTS])
                if parts["gate"]:
                    gate_events.append(
                        {
                            "stage_before": step["gate_stage"],
                            "step": step["step"],
                            "discounted_gate_bonus": parts["gate"] * GAMMA ** step["step"],
                        }
                    )
                previous = u
            parts = np.array(components)
            shaping = potential_terms(trace)
            factor = GAMMA ** np.arange(len(parts))
            error = abs(float(factor @ shaping) + potential(trace[0]["obs"]))
            assert error < 1e-8
            max_shape_error = max(max_shape_error, error)
            records[ep["eval_seed"]] = {
                "policy": policy,
                "seed": ep["eval_seed"],
                "success": ep["success"],
                "official_time": ep["official_time"],
                "raw_components": dict(zip(PARTS, parts.sum(axis=0).tolist())),
                "discounted_components": dict(zip(PARTS, (factor @ parts).tolist())),
                "discounted_components_by_stage": {
                    str(s): dict(
                        zip(
                            PARTS,
                            (
                                factor[np.array([x["gate_stage"] == s for x in trace])]
                                @ parts[np.array([x["gate_stage"] == s for x in trace])]
                            ).tolist(),
                        )
                    )
                    for s in range(6)
                },
                "gate_events": gate_events,
                "initial_potential": potential(trace[0]["obs"]),
                "progress_decomposition": progress_decomposition(trace, float(parts[:, 0].sum()))
                if ep["success"]
                else None,
                "variants": variant_scores(parts, shaping),
            }
        all_records[policy] = records
    assert all(set(r) == set(all_records["C0"]) for r in all_records.values())
    for seed in all_records["C0"]:
        assert (
            max(
                abs(
                    all_records[n][seed]["initial_potential"]
                    - all_records["C0"][seed]["initial_potential"]
                )
                for n in all_records
            )
            < 1e-8
        )
    comparisons = {
        "A4_vs_C0": compare_scores(all_records["C0"], all_records["A4"]),
        "A4_vs_C3": compare_scores(all_records["C3"], all_records["A4"]),
    }
    # Algebraic invariance is checked on successful, failed, and terminal-warped episodes.
    for comparison in comparisons.values():
        for name in ("completion_time_failure_only", "completion_time_failure_with_potential"):
            assert not comparison[name]["slower_scores_higher"]
            assert not comparison[name]["failure_scores_higher"]
    example = {name: all_records[name][110089] for name in ("C0", "A4")}
    result = {
        "gamma": GAMMA,
        "records": all_records,
        "comparisons": comparisons,
        "example_110089": example,
        "potential_telescope_max_error": max_shape_error,
        "scope": (
            "Offline scores on fixed trajectories, not newly trained policies "
            "or validated performance"
        ),
        "simulation_runs_new": 0,
        "training_performed": False,
        "validation_opened": False,
        "final_opened": False,
        "inputs_sha256": inputs,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output_dir / "reward_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "comparisons": comparisons,
                "example": example,
                "potential_telescope_max_error": max_shape_error,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
