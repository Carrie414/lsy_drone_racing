# Adaptive Level 2 controller

This controller is a tested reliability-oriented baseline, not a proof of optimal lap time or
100% success. It uses CPU simulation and needs no training, extra packages, or GPU.

## Run

From the repository root:

```powershell
pixi run python scripts/sim.py --config level2.toml
pixi run evaluate
```

The first command displays one randomized Level 2 flight. The second uses the unchanged course
20-episode evaluation script. `config/level2.toml` selects `adaptive_controller.py`; only its
`controller.file` field was changed. The original `state_controller.py` remains available.
No changes were pushed to GitHub or submitted to the course.

## How it works

- Follow the observed gate sequence and signed directions, including the final repeat:
  gate 1 -> 2 -> 3 -> 4 -> 2 (five crossings of four physical gates).
- Use nominal geometry until `gates_visited`/`obstacles_visited` reveals measured geometry.
  Replan when these public observation flags change. Never read hidden simulator state.
- Use a visibility graph and Dijkstra search around buffered obstacles and gate frames.
  Unknown geometry has larger buffers. The local graph approximates horizontal geometry;
  this is not a formally collision-free trajectory guarantee.
- Approach a sensed gate along its signed local x axis. Check the crossing segment against
  observed cylinders and select an exit, allowing a small lateral offset when necessary.
- Track waypoints with bounded position lookahead and velocity commands, extra velocity damping,
  and bounded vertical integral correction. Wait for entry alignment and sufficient exit clearance
  before changing phases. Return a 16-component state command to the existing low-level controller.
- If no route is found, hold position rather than treating the failed plan as an achieved entry.
  Some narrow randomized layouts can therefore time out. A fallback exploration/recovery planner
  is a potential improvement.

## Local validation (2026-10-28)

Repository base commit: `c576683`. Native Windows, existing Pixi CPU environment, original
Level 2 disturbances/randomizations, no rendering during evaluation.

| Evaluation | Success | Mean time of successful flights |
| --- | --- | --- |
| Development regression, seeds 201-220 | 17/20 (85%) | 23.546 s |
| Independent validation, seeds 301-320 | 18/20 (90%) | 23.309 s |
| Original example on the same seeds 301-320 | 1/20 (5%) | 20.920 s (one success) |
| Unmodified course `scripts/evaluate.py`, new random episodes | 19/20 (95%) | 23.901 s |

The independent validation failures were seed 304 (collision after one crossing) and seed 320
(timeout after three crossings). The course evaluation had one timeout after three crossings.
The independent validation's largest measured controller call was 15.191 ms on this machine,
within the 20 ms control period; this is not a real-time guarantee on other machines.
The course evaluation returned exit code 0 and wrote `evaluation.csv`.
Four targeted unit tests passed. Ruff checks passed for the new Python files.
Pytest reported only a cache-directory write warning; the tests themselves passed.

The improvement demonstrated here is completion reliability. The original baseline's one
successful flight was faster, so these results do not establish a lap-time speedup. Passing local
runs also does not guarantee success on the course server or acceptance against other teams.
No physical-drone deployment has been tested.

## Reproduce independent checks

```powershell
pixi run python tools/validate_adaptive.py --start-seed 301 --runs 20
pixi run python tools/validate_adaptive.py --controller state_controller.py --start-seed 301 --runs 20 --output logs/baseline-comparison.json
pixi run -e tests python -m pytest tests/unit/test_adaptive_controller.py -q
```

The validation tool uses the original environment's termination/timeout and five-crossing
completion rule. It does not change the official evaluator, track, randomization, or dynamics.
Detailed local run traces are in ignored `logs/`: `adaptive-holdout-final.json`,
`adaptive-v14.json`, `baseline-comparison.json`, and `adaptive-official-evaluation.txt`.
Keep this report when sharing the implementation; ignored logs are not included by Git.
