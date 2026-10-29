"""B0/B4: analyze public trajectories without guessing hidden collision causes."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from lsy_drone_racing.envs.race_core import build_action_space
from lsy_drone_racing.utils import load_config

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    out = args.run_dir
    rows = [json.loads(x) for x in (out / "episodes.jsonl").read_text().splitlines()]
    cfg = load_config(ROOT / "config/level2.toml")
    space = build_action_space("attitude", cfg.sim.drone, cfg.sim.dynamics)
    low, high = float(space.low[3]), float(space.high[3])

    def normalize(a):
        a = np.asarray(a)
        return np.array([a[0] / 1.5, a[1] / 1.5, a[2] / 0.5, 2 * (a[3] - low) / (high - low) - 1])

    failures = []
    recent_actionable = 0
    terminal_reveal = 0
    saturated = np.zeros(4, dtype=int)
    steps = 0
    transitions = defaultdict(list)
    for ep in rows:
        path = out / ep["trace_path"]
        with gzip.open(path, "rt") as f:
            trace = [json.loads(x) for x in f]
        prev = None
        previous_stage = None
        for x in trace:
            u = normalize(x["action"])
            saturated += np.abs(u) >= 0.99
            steps += 1
            if prev is not None and x["gate_stage"] != previous_stage:
                transitions[f"{previous_stage}->{x['gate_stage']}"].append(
                    float(np.linalg.norm(u - prev))
                )
            prev = u
            previous_stage = x["gate_stage"]
        if ep["success"]:
            continue
        last = trace[-1]
        obs = last["obs"]
        stage = last["gate_stage"]
        gate = obs["gate_sequence"][stage]
        recent = any(
            (not bool(x["obs"]["gates_visited"][gate]))
            and bool(x["next_obs"]["gates_visited"][gate])
            for x in trace[-10:]
        )
        known = bool(obs["gates_visited"][gate])
        recent_actionable += bool(known and recent)
        terminal_reveal += bool(not known and last["next_obs"]["gates_visited"][gate])
        failures.append(
            dict(
                seed=ep["eval_seed"],
                gate_stage=stage,
                official_time=ep["official_time"],
                target_gate_id=gate,
                target_revealed=bool(obs["gates_visited"][gate]),
                target_revealed_last_10_steps=recent,
                preterminal_speed=float(np.linalg.norm(obs["vel"])),
                preterminal_pos=obs["pos"],
                preterminal_velocity=obs["vel"],
                note=(
                    "Preterminal public observation. Unrevealed geometry is nominal; "
                    "collision object unknown."
                ),
            )
        )
    result = dict(
        episodes=len(rows),
        recorded_control_steps=steps,
        failures=len(failures),
        failure_target_unrevealed=sum(not x["target_revealed"] for x in failures),
        failure_target_revealed_last_10_steps=sum(
            x["target_revealed_last_10_steps"] for x in failures
        ),
        command_saturation_fraction_by_axis=dict(
            zip(["roll", "pitch", "yaw", "thrust"], (saturated / steps).tolist())
        ),
        saturation_definition=(
            "abs(normalized issued command)>=0.99, before environmental action noise"
        ),
        recent_reveal_available_before_terminal_action=recent_actionable,
        target_revealed_only_in_terminal_transition=terminal_reveal,
        normalized_stage_transition_jump={
            k: dict(n=len(a), p50=float(np.median(a)), p90=float(np.percentile(a, 90)), max=max(a))
            for k, a in transitions.items()
        },
        interpretations=(
            "Descriptive diagnostics only; no collision-cause or intervention-effect claim."
        ),
    )
    (out / "failure_diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "failure_cases.json").write_text(json.dumps(failures, indent=2) + "\n")
    with (out / "failure_cases.tsv").open("w") as f:
        fields = [
            "seed",
            "gate_stage",
            "official_time",
            "target_gate_id",
            "target_revealed",
            "target_revealed_last_10_steps",
            "preterminal_speed",
        ]
        w = csv.DictWriter(f, fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(failures)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
