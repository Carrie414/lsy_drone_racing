"""A-side first-gate diagnosis using recorded public observations only."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def geometry(obs: dict, gate: int) -> dict:
    """Report observed gate-frame geometry and a constant-velocity projection.

    This projection is a diagnostic, not a collision detector or dynamics model.
    Geometry before revelation is nominal and must not be treated as truth.
    """
    frame = Rotation.from_quat(obs["gates_quat"][gate]).as_matrix()
    local = frame.T @ (np.asarray(obs["pos"]) - obs["gates_pos"][gate])
    velocity = frame.T @ np.asarray(obs["vel"])
    eta = -local[0] / velocity[0] if abs(velocity[0]) > 1e-6 else -1.0
    projected = local + eta * velocity if 0 <= eta <= 1 else None
    return {
        "normal_distance_m": float(local[0]),
        "lateral_m": float(local[1]),
        "vertical_m": float(local[2]),
        "speed_mps": float(np.linalg.norm(obs["vel"])),
        "plane_eta_s": float(eta) if 0 <= eta <= 1 else None,
        "projected_lateral_m": float(projected[1]) if projected is not None else None,
        "projected_vertical_m": float(projected[2]) if projected is not None else None,
    }


def summarize_episode(episode: dict, trace: list[dict]) -> dict:
    """Measure first-gate passage, revelation and available correction time."""
    gate = int(trace[0]["obs"]["gate_sequence"][0])
    first = [step for step in trace if int(step["gate_stage"]) == 0]
    end = first[-1]
    revealed = next((step for step in first if bool(step["obs"]["gates_visited"][gate])), None)
    reveal_transition = next(
        (
            step
            for step in first
            if not bool(step["obs"]["gates_visited"][gate])
            and bool(step["next_obs"]["gates_visited"][gate])
        ),
        None,
    )
    passed = int(end["next_obs"]["n_gates_passed"]) >= 1
    row = {
        "seed": episode["eval_seed"],
        "race_success": episode["success"],
        "first_gate_passed": passed,
        "first_stage_seconds": (end["step"] + 1) / 50,
        "reveal_available_step": revealed["step"] if revealed else None,
        "correction_steps": end["step"] - revealed["step"] + 1 if revealed else 0,
        "terminal_only_reveal": bool(reveal_transition is not None and revealed is None),
        "reveal_position_jump_m": None,
        "reveal_command_jump_scaled": None,
    }
    if reveal_transition is not None:
        row["reveal_position_jump_m"] = float(
            np.linalg.norm(
                np.asarray(reveal_transition["next_obs"]["gates_pos"][gate])
                - reveal_transition["obs"]["gates_pos"][gate]
            )
        )
    if revealed is not None:
        row.update(
            {f"reveal_{key}": value for key, value in geometry(revealed["obs"], gate).items()}
        )
        index = first.index(revealed)
        if index:
            # Radians for attitudes, newtons for thrust: explicitly scaled, not physical norm.
            delta = np.asarray(revealed["action"]) - first[index - 1]["action"]
            row["reveal_command_jump_scaled"] = float(
                np.linalg.norm(delta / np.array([1.5, 1.5, 0.5, 1.0]))
            )
    row.update({f"last_{key}": value for key, value in geometry(end["obs"], gate).items()})
    row["last_geometry_revealed"] = bool(end["obs"]["gates_visited"][gate])
    return row


def distribution(rows: list[dict], field: str) -> dict:
    """Describe available values without silently counting missing values as zero."""
    values = [row[field] for row in rows if row.get(field) is not None]
    return {
        "n": len(values),
        "median": float(np.median(values)) if values else None,
        "p90": float(np.percentile(values, 90)) if values else None,
    }


def main() -> None:
    """Write per-seed findings and explicitly noncausal group comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    episodes = [
        json.loads(line) for line in (args.run_dir / "episodes.jsonl").read_text().splitlines()
    ]
    rows = []
    for episode in episodes:
        with gzip.open(args.run_dir / episode["trace_path"], "rt", encoding="utf-8") as stream:
            rows.append(summarize_episode(episode, [json.loads(line) for line in stream]))
    metrics = [
        "first_stage_seconds",
        "correction_steps",
        "reveal_position_jump_m",
        "reveal_command_jump_scaled",
        "reveal_speed_mps",
        "reveal_plane_eta_s",
        "reveal_projected_lateral_m",
        "reveal_projected_vertical_m",
    ]
    summary = {}
    for passed in (False, True):
        group = [row for row in rows if row["first_gate_passed"] == passed]
        summary["failed_first_gate" if not passed else "passed_first_gate"] = {
            "n": len(group),
            "seeds": [row["seed"] for row in group],
            "unrevealed_at_last_action": sum(not row["last_geometry_revealed"] for row in group),
            "terminal_only_reveal": sum(row["terminal_only_reveal"] for row in group),
            "metrics": {key: distribution(group, key) for key in metrics},
        }
    summary["limitations"] = (
        "Public observations only. Before revelation geometry is nominal. Constant-velocity "
        "projection ignores control and acceleration; it does not identify collision objects "
        "or prove why an episode failed. Development seeds only; no held-out evaluation."
    )
    (args.run_dir / "first_gate_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.run_dir / "first_gate_cases.json").write_text(json.dumps(rows, indent=2) + "\n")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (args.run_dir / "first_gate_cases.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
