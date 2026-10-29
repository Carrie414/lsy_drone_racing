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
    base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    original_model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
    plan_version="2026-10-29-v2",
    contract_version="v2",
    required_a_files={p: (ROOT / p).exists() for p in expected},
    training_started=False,
    final_set_opened=False,
    candidate_evaluation_performed=False,
)
for p in ["torch", "pytest"]:
    try:
        result[p] = importlib.metadata.version(p)
    except importlib.metadata.PackageNotFoundError:
        result[p] = "not installed in active default Pixi environment"
result["blockers"] = [p for p, present in result["required_a_files"].items() if not present]
result["blockers"].append("No A candidate weight / exported manifest for C1-C5")
print(json.dumps(result, ensure_ascii=False, indent=2))
