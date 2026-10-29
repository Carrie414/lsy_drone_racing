"""B0: original Level 2 evaluation with full provenance and public-observation traces."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import gymnasium
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.utils import load_config, load_controller

ROOT = Path(__file__).resolve().parents[2]


def git_output(arguments: list[str], *, text: bool = True) -> str | bytes:
    """Query this exact repository without changing global safe.directory settings."""
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", *arguments], cwd=ROOT, text=text
    )


def sha(path: Path | str) -> str:
    """Hash a recorded source, model or config file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_obs(obs: dict) -> dict:
    """Serialize only the public observation without simulator internals."""
    return {k: np.asarray(v).tolist() for k, v in obs.items()}


def wilson(k: int, n: int) -> list[float]:
    """Return the binomial success-rate Wilson 95 percent interval."""
    z = 1.95996398454
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / den
    return [c - h, c + h]


def validate_split(seed: int, runs: int, split: str) -> None:
    """Require explicit frozen validation and keep the final set closed."""
    seeds = set(range(seed, seed + runs))
    validation = set(range(120001, 120101))
    if runs < 1 or split not in {"development", "validation"}:
        raise ValueError("Positive runs and a declared development/validation split required")
    if seeds & set(range(130001, 130501)):
        raise ValueError(
            "Final set is reserved; use a separately reviewed frozen-candidate protocol"
        )
    if split == "validation" and not seeds <= validation:
        raise ValueError("Validation must stay within frozen seeds 120001-120100")
    if split == "development" and seeds & validation:
        raise ValueError("Reserved validation requires explicit --split validation")


def main() -> None:
    """Evaluate a candidate on the explicitly selected non-final seed range."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", default="fast_racing_controller.py")
    parser.add_argument("--seed", type=int, default=110001)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--split", choices=["development", "validation"], default="development")
    parser.add_argument("--run-id", default="C0-development-20261029")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--output-root", type=Path, default=ROOT / "experiments/team_runs")
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--artifact-manifest", type=Path)
    args = parser.parse_args()
    if args.runs < 1 or Path(args.run_id).name != args.run_id:
        parser.error("Positive run count and simple run_id required")
    try:
        validate_split(args.seed, args.runs, args.split)
    except ValueError as error:
        parser.error(str(error))
    out = args.output_root / args.run_id
    if out.exists():
        parser.error(f"Refusing to overwrite {out}")
    out.mkdir(parents=True)
    config = load_config(ROOT / "config/level2.toml")
    config.sim.render = False
    path = ROOT / "lsy_drone_racing/control" / args.controller
    cls = load_controller(path)
    original_model = ROOT / "lsy_drone_racing/control/policies/level2_reference.npz"
    model = original_model
    if (args.artifact is None) != (args.artifact_manifest is None):
        parser.error("--artifact and --artifact-manifest must be provided together")
    if args.artifact is not None:
        if args.controller != "team_policy_controller.py":
            parser.error("External artifacts require the verified team_policy_controller.py")
        artifact_manifest = json.loads(args.artifact_manifest.read_text(encoding="utf-8"))
        if sha(args.artifact) != artifact_manifest["artifact_sha256"]:
            parser.error("Candidate model hash does not match its manifest")
        model = args.artifact.resolve()
        cls = type(
            "EvaluationTeamPolicy",
            (cls,),
            {
                "POLICY_FILE": str(model),
                "MANIFEST_FILE": str(args.artifact_manifest.resolve()),
                "POLICY_SHA256": artifact_manifest["artifact_sha256"],
            },
        )
    elif hasattr(cls, "MANIFEST_FILE"):
        model = ROOT / "lsy_drone_racing/control/policies_team" / cls.POLICY_FILE
    tracked = git_output(["ls-files"]).splitlines()
    original = {name: sha(ROOT / name) for name in tracked if (ROOT / name).is_file()}
    manifest = dict(
        run_id=args.run_id,
        candidate="C0" if args.controller == "fast_racing_controller.py" else "candidate",
        contract_version="v2",
        config="level2.toml",
        controller=args.controller,
        commit=git_output(["rev-parse", "HEAD"]).strip(),
        dirty_status=git_output(["status", "--short", "--untracked-files=all"]),
        tracked_diff_sha256=hashlib.sha256(
            git_output(["diff", "HEAD", "--binary"], text=False)
        ).hexdigest(),
        controller_sha256=sha(path),
        model_sha256=sha(model),
        model_path=str(model),
        original_model_sha256=sha(original_model),
        artifact_manifest_sha256=sha(args.artifact_manifest) if args.artifact_manifest else None,
        config_sha256=sha(ROOT / "config/level2.toml"),
        evaluator_sha256=sha(__file__),
        original_tracked_sha256=original,
        executable=sys.executable,
        prefix=sys.prefix,
        python=sys.version,
        platform=platform.platform(),
        machine=platform.machine(),
        cpu_count=os.cpu_count(),
        runtime_mode=(
            "Direct existing Pixi environment interpreter, not venv; "
            "pixi activation in agent sandbox unavailable"
        ),
        versions={
            p: importlib.metadata.version(p)
            for p in ["numpy", "jax", "jaxlib", "mujoco", "crazyflow", "gymnasium", "scipy"]
        },
        seeds=list(range(args.seed, args.seed + args.runs)),
        split=args.split,
        purpose=(
            "frozen-candidate validation; final set remains unopened"
            if args.split == "validation"
            else "development baseline; validation/final sets remain unopened"
        ),
        environment_overrides={"sim.render": False},
        time_convention=(
            "official_time=zero-based final step index/50; "
            "elapsed_steps_time=number of control steps/50"
        ),
        script_environment={
            k: os.environ.get(k)
            for k in ["SCIPY_ARRAY_API", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "PYTHONPATH"]
        },
    )
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    env = JaxToNumpy(
        gymnasium.make(
            config.env.id,
            freq=config.env.freq,
            sim_config=config.sim,
            sensor_range=config.env.sensor_range,
            control_mode=config.env.control_mode,
            track=config.env.track,
            disturbances=config.env.get("disturbances"),
            randomizations=config.env.get("randomizations"),
            seed=args.seed,
        )
    )
    rows = []
    all_compute = []
    all_callback = []
    all_total = []
    try:
        for seed in manifest["seeds"]:
            tic = time.perf_counter()
            obs, info = env.reset(seed=seed)
            reset_ms = (time.perf_counter() - tic) * 1000
            tic = time.perf_counter()
            ctl = cls(obs, info, config)
            init_ms = (time.perf_counter() - tic) * 1000
            measures = []
            last_action = None
            last_stage = None
            transitions = []
            visited_updates = []
            trace = (
                gzip.open(out / f"trace-{seed}.jsonl.gz", "wt", encoding="utf-8")
                if args.trace
                else None
            )
            try:
                for step in range(100000):
                    before = json_obs(obs) if args.trace else None
                    stage = int(obs["n_gates_passed"])
                    vis = np.asarray(obs["gates_visited"]).copy()
                    tic = time.perf_counter()
                    action = ctl.compute_control(obs, info)
                    compute = time.perf_counter() - tic
                    if (
                        action.shape != (4,)
                        or not np.isfinite(action).all()
                        or not env.action_space.contains(action)
                    ):
                        raise ValueError("Invalid or out-of-bounds action")
                    tic = time.perf_counter()
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    step_ms = (time.perf_counter() - tic) * 1000
                    tic = time.perf_counter()
                    finished = ctl.step_callback(
                        action, next_obs, reward, terminated, truncated, info
                    )
                    callback = time.perf_counter() - tic
                    total = compute + callback
                    measures.append([compute, callback, total, step_ms])
                    all_compute.append(compute * 1000)
                    all_callback.append(callback * 1000)
                    all_total.append(total * 1000)
                    # Stage changed by the previous environment step: compare its adjacent commands.
                    if last_action is not None and stage != last_stage:
                        transitions.append(
                            dict(
                                step=step,
                                from_stage=last_stage,
                                to_stage=stage,
                                command_delta=(action - last_action).tolist(),
                                delta_norm=float(np.linalg.norm(action - last_action)),
                                pos=np.asarray(obs["pos"]).tolist(),
                                vel=np.asarray(obs["vel"]).tolist(),
                            )
                        )
                    if np.any(np.asarray(next_obs["gates_visited"]) != vis):
                        visited_updates.append(
                            dict(
                                step=step,
                                new_ids=np.flatnonzero(
                                    np.asarray(next_obs["gates_visited"]) & ~vis
                                ).tolist(),
                            )
                        )
                    if trace is not None:
                        trace.write(
                            json.dumps(
                                dict(
                                    seed=seed,
                                    step=step,
                                    gate_stage=stage,
                                    obs=before,
                                    action=action.tolist(),
                                    compute_ms=compute * 1000,
                                    callback_ms=callback * 1000,
                                    total_control_ms=total * 1000,
                                    environment_step_ms=step_ms,
                                    official_reward=float(reward),
                                    next_obs=json_obs(next_obs),
                                    terminated=bool(terminated),
                                    truncated=bool(truncated),
                                    controller_finished=bool(finished),
                                )
                            )
                            + "\n"
                        )
                    last_action = action.copy()
                    last_stage = stage
                    obs = next_obs
                    if terminated or truncated or finished:
                        break
            finally:
                if trace is not None:
                    trace.close()
            success = bool(obs["n_gates_passed"] == len(obs["gate_sequence"]))
            arr = np.asarray(measures)
            control_ms = arr[:, 2] * 1000
            row = dict(
                run_id=args.run_id,
                eval_seed=seed,
                training_seed=None,
                success=success,
                gate_pass_count=int(obs["n_gates_passed"]),
                official_time=step / config.env.freq,
                elapsed_steps_time=(step + 1) / config.env.freq,
                control_steps=step + 1,
                terminated=bool(terminated),
                truncated=bool(truncated),
                controller_finished=bool(finished),
                failure_label=None
                if success
                else ("task_timeout" if truncated else "terminal_non_success"),
                init_ms=init_ms,
                reset_ms=reset_ms,
                first_control_ms=float(control_ms[0]),
                first_environment_step_ms=float(arr[0, 3]),
                compute_p95_ms=float(np.percentile(arr[:, 0] * 1000, 95)),
                callback_p95_ms=float(np.percentile(arr[:, 1] * 1000, 95)),
                total_p95_ms=float(np.percentile(control_ms, 95)),
                total_max_ms=float(max(control_ms)),
                over_budget_steps=int(np.sum(control_ms > 1000 / config.env.freq)),
                transition_commands=transitions,
                gate_observation_updates=visited_updates,
                trace_path=f"trace-{seed}.jsonl.gz" if args.trace else None,
            )
            rows.append(row)
            with (out / "episodes.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            ctl.episode_callback()
            ctl.episode_reset()
    finally:
        env.close()
    ok = [r["official_time"] for r in rows if r["success"]]
    timings = {}
    for name, a in [
        ("compute_control", all_compute),
        ("step_callback", all_callback),
        ("total", all_total),
    ]:
        timings[name] = {f"p{q}_ms": float(np.percentile(a, q)) for q in [50, 95, 99]}
        timings[name]["max_ms"] = float(max(a))
    summary = dict(
        run_id=args.run_id,
        runs=len(rows),
        successes=len(ok),
        success_rate=len(ok) / len(rows),
        success_rate_wilson95=wilson(len(ok), len(rows)),
        mean_successful_seconds=float(np.mean(ok)) if ok else None,
        median_successful_seconds=float(np.median(ok)) if ok else None,
        p90_successful_seconds=float(np.percentile(ok, 90)) if ok else None,
        failed_gate_stages=dict(Counter(r["gate_pass_count"] for r in rows if not r["success"])),
        failed_seeds=[r["eval_seed"] for r in rows if not r["success"]],
        over_budget_steps=sum(r["over_budget_steps"] for r in rows),
        over_budget_episodes=sum(r["over_budget_steps"] > 0 for r in rows),
        timings=timings,
        all_original_tracked_files_unchanged=all(sha(ROOT / f) == h for f, h in original.items()),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
