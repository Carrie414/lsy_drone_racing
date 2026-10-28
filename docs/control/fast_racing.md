# Fast Level 2 racing policy

This is an attributed reproduction of a public, externally trained racing policy, not a policy newly trained by this team. It is validated for Level 2 only.

## Provenance and license

- Original actor: https://github.com/Xiaolong-Li1904/lsy_drone_racing/tree/1cca34f2476d7791bdb7074fc9cf3dc32b4fbc27
- NumPy adapter: https://github.com/mahdisafari761/lsy_drone_racing/tree/183c7f6850a29905a66c846e4b7d0bdd0a668d11
- Original files: rl/features.py, rl/actions.py, rl/policy_np.py, control/level2_rl_controller.py; adapted via control/level2_reference.py.
- Full MIT license: lsy_drone_racing/control/policies/level2_reference_LICENSE.
- Frozen actor SHA-256: d15e6d613041907ed5baa565f87ee3393df099b78ecb53b4428440399baede9a.

The local entry adds checksum verification and guards against accidental use with fully randomized Level 3 tracks. The reference adapter preserves the public policy's behavior. The observation encoder uses public sensor observations, signed gate sequences, repeated gates and previous actions. Arrays load with allow_pickle=False. No hidden simulator state or race seed is used by the controller.

## Validation

- Original official evaluate.py: 17/20 successful races, mean successful time 3.350588 seconds.
- Separate frozen-seed Level 2 evaluation, seeds 701-800: 81/100 successes, mean successful time 3.359506 seconds; all successful times below 10 seconds.
- Six unit regressions pass, including signed/repeated gates and action equality with the reproduced adapter.
- No guarantee of success on every random race or of Level 3 performance.

Times use the original evaluator's zero-based tick convention and average successful races only. Different random evaluations will vary.

## Usage

config/level2.toml selects fast_racing_controller.py and control_mode = attitude. After installing the normal course dependencies:

    pixi run python scripts/sim.py --config=level2.toml --render=True --n_runs=20
    pixi run evaluate

On Windows, Ctrl+Shift+B runs the included VS Code visualization task using the existing Pixi Python environment. If an input method intercepts the shortcut, use Terminal > Run Task > Level 2: visualize 20 races. See LEVEL2_QUICKSTART.md.

Only the allowed controller filename and control mode fields change in the race configuration. Official simulation/evaluation scripts, environment, physics, sensor range, noise, randomization, scoring and timeout are unchanged. The older adaptive controller remains available in the repository and in commit 83d1fc7.