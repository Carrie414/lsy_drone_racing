"""Report readiness without inventing absent A modules, weights, or training runs."""

import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
expected = [
    "training/team_contract.py",
    "training/team_a/policy.py",
    "training/team_a/ppo_update.py",
    "training/team_a/export_npz.py",
    "lsy_drone_racing/control/team_policy_controller.py",
]
model = ROOT / "lsy_drone_racing/control/policies/level2_reference.npz"
result = dict(
    base_commit=subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip(),
    original_model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
    plan_version="2026-10-29-v2",
    contract_version="v2",
    required_a_files={p: (ROOT / p).exists() for p in expected},
    final_set_opened=None,
    candidate_evaluation_performed=None,
    scope="Local source and asset readiness only; inspect run manifests for evaluation history",
)
exports = []
for path in sorted((ROOT / "lsy_drone_racing/control/policies_team").glob("*.manifest.json")):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    asset = path.with_name(path.name.removesuffix(".manifest.json") + ".npz")
    if (
        manifest.get("training_manifest", {}).get("training_kind") == "racing"
        and manifest.get("update_count", 0) > 0
        and asset.is_file()
        and hashlib.sha256(asset.read_bytes()).hexdigest() == manifest.get("artifact_sha256")
    ):
        exports.append(
            {
                "model": str(asset.relative_to(ROOT)),
                "profile": manifest["profile"],
                "artifact_sha256": manifest["artifact_sha256"],
            }
        )
result["racing_export_evidence"] = exports
result["training_started"] = bool(exports)
for p in ["torch", "pytest"]:
    try:
        result[p] = importlib.metadata.version(p)
    except importlib.metadata.PackageNotFoundError:
        result[p] = "not installed in active default Pixi environment"
result["blockers"] = [p for p, present in result["required_a_files"].items() if not present]
if not exports:
    result["blockers"].append("No hash-matched trained racing export found locally")
print(json.dumps(result, ensure_ascii=False, indent=2))
