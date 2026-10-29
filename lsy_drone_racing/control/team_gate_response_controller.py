"""Experimental bounded first-gate revelation response on the pinned C3 actor.

Only public old/new gate observations are compared. This is a deterministic
controller-side experiment, not a newly trained model or a default replacement.
"""

from __future__ import annotations

import numpy as np

from lsy_drone_racing.control.level2_reference import reference_features
from lsy_drone_racing.control.team_c3_pilot_controller import PilotController


class GateResponseController(PilotController):
    """Temporarily amplify a gate-update response with bounded command residuals."""

    RESPONSE_GAIN = 0.25
    RESPONSE_STEPS = 5
    RESIDUAL_BOUND = 0.08

    def reset(self) -> None:
        """Discard gate memory and transient residual state between episodes."""
        super().reset()
        self._last_gate = None
        self._gate_before_reveal = None
        self._response_age = self.RESPONSE_STEPS

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        """Compare revealed and previously nominal geometry at the same state."""
        stage = int(obs["n_gates_passed"])
        if stage != 0:
            self._gate_before_reveal = None
            self._response_age = self.RESPONSE_STEPS
        gate = int(obs["gate_sequence"][0])
        current = (
            np.asarray(obs["gates_pos"][gate]).copy(),
            np.asarray(obs["gates_quat"][gate]).copy(),
            bool(obs["gates_visited"][gate]),
        )
        if stage == 0 and self._last_gate is not None:
            if not self._last_gate[2] and current[2]:
                self._gate_before_reveal = self._last_gate
                self._response_age = 0
        previous = self.previous.copy()
        action = super().compute_control(obs, info)
        if self._gate_before_reveal is not None and self._response_age < self.RESPONSE_STEPS:
            nominal = dict(obs)
            for key, old in zip(("gates_pos", "gates_quat"), self._gate_before_reveal[:2]):
                nominal[key] = np.array(obs[key], copy=True)
                nominal[key][gate] = old
            # Preserve the visited flag: isolate geometry-response, not categorical flag effects.
            x = reference_features(nominal, previous)[None]
            alternative = self.policy.commands(x, np.array([0], dtype=np.int64))[0]
            base = self.previous.copy()
            fade = 1 - self._response_age / self.RESPONSE_STEPS
            residual = np.clip(
                self.RESPONSE_GAIN * fade * (base - alternative),
                -self.RESIDUAL_BOUND,
                self.RESIDUAL_BOUND,
            )
            command = np.clip(base + residual, -1, 1).astype(np.float64)
            action = np.array(
                [
                    command[0] * 1.5,
                    command[1] * 1.5,
                    command[2] * 0.5,
                    self.thrust_low + 0.5 * (command[3] + 1) * (self.thrust_high - self.thrust_low),
                ],
                dtype=np.float32,
            )
            action = np.clip(action, self.space.low, self.space.high).astype(np.float32)
            self._remember_action(action)
            self._response_age += 1
        self._last_gate = current
        return action
