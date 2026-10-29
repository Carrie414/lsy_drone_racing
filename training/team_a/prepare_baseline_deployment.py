"""Export an unchanged baseline into the team controller's package format."""

from __future__ import annotations

import json
from pathlib import Path

from training.team_a.engine import PPOSettings, StagePPOEngine


def main() -> None:
    """Create the initial control asset once; never train or overwrite artifacts."""
    root = Path(__file__).resolve().parents[2]
    asset = root / "lsy_drone_racing/control/policies_team/team_policy_baseline.npz"
    manifest_path = asset.with_suffix(".manifest.json")
    if asset.exists() or manifest_path.exists():
        raise FileExistsError("The baseline package already exists; do not regenerate in place")
    engine = StagePPOEngine(PPOSettings())
    manifest = engine.export_npz(
        asset,
        {
            "run_id": "baseline-format-conversion-no-training",
            "policy_commit": "47733fa",
            "training_commit": "none-no-training",
            "training_kind": "baseline-conversion",
        },
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"asset": str(asset), "sha256": manifest["artifact_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
