"""Compare same-machine pilot evaluations on the same development seed set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_run(path: Path) -> tuple[dict, dict]:
    """Read preserved raw episode outcomes and their original evaluator summary."""
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (path / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    by_seed = {row["eval_seed"]: row for row in rows}
    if len(rows) != len(by_seed) or len(rows) != summary["runs"]:
        raise ValueError("Duplicate seeds or incomplete evaluation records")
    return summary, by_seed


def compare(root: Path) -> dict:
    """Apply the pilot's preregistered Pareto screen without opening held-out sets."""
    baseline, original = load_run(root / "C0-windows-development")
    outcomes = {}
    names = ["C1", "C2"]
    if (root / "C3-windows-development/summary.json").exists():
        names.append("C3")
    for name in names:
        summary, candidate = load_run(root / f"{name}-windows-development")
        if set(original) != set(candidate):
            raise ValueError("Candidate and baseline evaluation seeds differ")
        common = [s for s in original if original[s]["success"] and candidate[s]["success"]]
        differences = [candidate[s]["official_time"] - original[s]["official_time"] for s in common]
        improved = sum(candidate[s]["success"] and not original[s]["success"] for s in original)
        lost = sum(original[s]["success"] and not candidate[s]["success"] for s in original)
        eligible = (
            summary["successes"] >= baseline["successes"]
            and summary["mean_successful_seconds"] is not None
            and summary["mean_successful_seconds"] <= baseline["mean_successful_seconds"]
            and (
                summary["successes"] > baseline["successes"]
                or summary["mean_successful_seconds"] < baseline["mean_successful_seconds"]
            )
        )
        outcomes[name] = {
            "summary": summary,
            "common_successes": len(common),
            "common_success_mean_time_change": float(np.mean(differences)) if differences else None,
            "common_success_median_time_change": float(np.median(differences))
            if differences
            else None,
            "baseline_fail_candidate_success": improved,
            "baseline_success_candidate_fail": lost,
            "eligible_for_further_validation": eligible,
        }
    eligible = [
        name for name, result in outcomes.items() if result["eligible_for_further_validation"]
    ]
    ranked = sorted(
        eligible,
        key=lambda name: (
            -outcomes[name]["summary"]["successes"],
            outcomes[name]["summary"]["mean_successful_seconds"],
        ),
    )
    report = {
        "status": "DEVELOPMENT_PILOT_COMPARISON_COMPLETE",
        "baseline": baseline,
        "candidates": outcomes,
        "worth_further_validation": ranked,
        "default_formal_controller_changed": False,
        "validation_opened": False,
        "final_opened": False,
        "limitations": "Single-seed development pilot; no final reliability claim.",
    }
    (root / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    """Write the reproducible development comparison next to the original runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
