"""Install an explicitly named racing pilot without changing the default controller."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from lsy_drone_racing.control.team_policy_runtime import NumpyStagePolicy
from training.team_a.npz_io import file_sha256


def package_candidate(asset: Path, manifest_path: Path, name: str) -> Path:
    """Copy a validated artifact and create one official-loader-compatible subclass."""
    if re.fullmatch(r"team_[a-z0-9_]{1,48}", name) is None:
        raise ValueError("Use a simple lowercase name starting with team_")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    policy = NumpyStagePolicy(asset, manifest["artifact_sha256"])
    if policy.training_kind != "racing" or policy.update_count < 1:
        raise ValueError("Only genuinely updated racing artifacts may be packaged")
    if manifest["training_manifest"]["training_kind"] != "racing":
        raise ValueError("Manifest does not describe racing training")
    root = Path(__file__).resolve().parents[2] / "lsy_drone_racing/control"
    destination = root / "policies_team" / f"{name}.npz"
    manifest_destination = destination.with_suffix(".manifest.json")
    controller = root / f"{name}_controller.py"
    if any(p.exists() for p in [destination, manifest_destination, controller]):
        raise FileExistsError("Use a new candidate name; existing artifacts are immutable")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(asset, destination)
    shutil.copy2(manifest_path, manifest_destination)
    assert file_sha256(destination) == manifest["artifact_sha256"]
    source = (
        '"""Explicit pilot entry; performance must be read with its development report."""\n\n'
        "from lsy_drone_racing.control.team_policy_controller import TeamPolicyController\n\n\n"
        "class PilotController(TeamPolicyController):\n"
        '    """Load this pilot while preserving the default configuration."""\n\n'
        f'    POLICY_FILE = "{destination.name}"\n'
        f'    MANIFEST_FILE = "{manifest_destination.name}"\n'
        f'    POLICY_SHA256 = "{manifest["artifact_sha256"]}"\n'
    )
    controller.write_text(source, encoding="utf-8")
    return controller


def main() -> None:
    """Package one selected pilot, preserving all existing model and config files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    print(package_candidate(args.artifact, args.manifest, args.name))


if __name__ == "__main__":
    main()
