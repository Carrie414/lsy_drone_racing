"""Team NumPy controller entry, initially using an unchanged baseline conversion.

This initial package is a deployment-chain check, not a faster trained policy.
The original actor is credited in policies_team/level2_reference_LICENSE and the
asset manifest. Formal evaluation configuration remains untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.level2_reference import reference_features
from lsy_drone_racing.control.team_policy_runtime import NumpyStagePolicy
from lsy_drone_racing.envs.race_core import build_action_space


class TeamPolicyController(Controller):
    """Load a pinned two-actor profile and preserve the public attitude interface."""

    POLICY_FILE = "team_policy_baseline.npz"
    MANIFEST_FILE = "team_policy_baseline.manifest.json"
    POLICY_SHA256 = "e965ad393e1f96442653505d0c507705cfd52960ea4b250a27694d61b40abf05"

    def __init__(self, obs: dict, info: dict, config: dict) -> None:
        """Validate interface, provenance and asset integrity before the first step."""
        super().__init__(obs, info, config)
        if config.env.control_mode != "attitude":
            raise ValueError("TeamPolicyController requires the 4D attitude interface")
        if config.env.freq != 50:
            raise ValueError("This profile is validated only at 50 Hz")
        if config.env.track.randomize:
            raise ValueError("This Level 2 profile is not validated for Level 3 random tracks")
        asset_root = Path(__file__).parent / "policies_team"
        self.policy = NumpyStagePolicy(asset_root / self.POLICY_FILE, self.POLICY_SHA256)
        manifest = json.loads((asset_root / self.MANIFEST_FILE).read_text(encoding="utf-8"))
        if (
            manifest["artifact_sha256"] != self.POLICY_SHA256
            or manifest["profile"] != self.policy.profile
            or manifest["active_stages"] != self.policy.active_stages.tolist()
            or manifest["n_stages"] != self.policy.n_stages
            or manifest["training_manifest"]["training_kind"] != self.policy.training_kind
            or manifest["update_count"] != self.policy.update_count
        ):
            raise ValueError("Manifest does not describe the selected model")
        if self.policy.training_kind not in {"baseline-conversion", "racing"}:
            raise ValueError("Synthetic or unknown test weights cannot be deployed for racing")
        if self.policy.training_kind == "baseline-conversion" and self.policy.update_count != 0:
            raise ValueError("Baseline conversion must have zero training updates")
        if len(obs["gate_sequence"]) + 1 != self.policy.n_stages:
            raise ValueError("Profile gate count does not match the public gate sequence")
        self.space = build_action_space("attitude", config.sim.drone, config.sim.dynamics)
        self.thrust_low, self.thrust_high = float(self.space.low[3]), float(self.space.high[3])
        self.reset()

    def _remember_action(self, action: np.ndarray) -> None:
        """Store the final issued command, before the environment's own action noise."""
        command = np.asarray(action, dtype=np.float64)
        if command.shape != (4,) or not np.isfinite(command).all():
            raise ValueError("Final attitude command must be finite and have shape [4]")
        normalized = command.copy()
        normalized[:2] /= 1.5
        normalized[2] /= 0.5
        normalized[3] = (
            2 * (command[3] - self.thrust_low) / (self.thrust_high - self.thrust_low) - 1
        )
        self.previous = np.clip(normalized, -1, 1).astype(np.float32)

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        """Compute one bounded [roll, pitch, yaw, total thrust] command."""
        if len(obs["gate_sequence"]) + 1 != self.policy.n_stages:
            raise ValueError("Public gate sequence changed to an unsupported length")
        value = reference_features(obs, self.previous)[None, :]
        stage = np.array([obs["n_gates_passed"]], dtype=np.int64)
        u = self.policy.commands(value, stage)[0].astype(np.float64)
        action = np.array(
            [
                u[0] * 1.5,
                u[1] * 1.5,
                u[2] * 0.5,
                self.thrust_low + 0.5 * (u[3] + 1) * (self.thrust_high - self.thrust_low),
            ],
            dtype=np.float32,
        )
        action = np.clip(action, self.space.low, self.space.high).astype(np.float32)
        if not np.isfinite(action).all():
            raise ValueError("Policy produced an invalid command")
        self._remember_action(action)
        return action

    def step_callback(
        self,
        action: np.ndarray,
        obs: dict,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Record the caller's final command and leave completion to the environment."""
        self._remember_action(action)
        return False

    def reset(self) -> None:
        """Clear this controller instance's command history."""
        self.previous = np.zeros(4, dtype=np.float32)

    def episode_reset(self) -> None:
        """Support official runner resets and reused controller instances."""
        self.reset()

    def episode_callback(self) -> None:
        """Clear history at the end of an episode without performing online training."""
        self.reset()
