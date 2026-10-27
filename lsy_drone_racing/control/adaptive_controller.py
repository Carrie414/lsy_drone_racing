"""Observation-driven gate alignment with collision-buffered planning and velocity feedback."""

from __future__ import annotations

import heapq
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class AdaptiveController(Controller):
    """Replan from sensed gate poses, then cross along each gate's local x axis."""

    def __init__(self, obs: dict, info: dict, config: dict):
        """Initialize a fresh flight using only the public observation and race configuration."""
        super().__init__(obs, info, config)
        self.dt = 1.0 / config.env.freq
        self.phase = "takeoff"
        self.index = 0
        self.route = []
        self.start = np.array(obs["pos"], dtype=float)
        self.goal = self.start.copy()
        self.goal[2] = 0.65
        self.known = False
        self.bias = np.zeros(3)
        self.seen = None
        self.last_setpoint = self.start.copy()
        self.velocity = np.zeros(3)

    def _set_target(self, goal: NDArray):
        """Set the next position for the bounded, damped velocity tracker."""
        self.goal = np.array(goal, dtype=float)

    def _geometry(self, obs: dict) -> tuple[int, NDArray, NDArray]:
        """Read the requested gate and signed crossing axis from the observation."""
        gate = int(obs["gate_sequence"][self.index])
        center = np.array(obs["gates_pos"][gate], dtype=float)
        normal = Rotation.from_quat(obs["gates_quat"][gate]).apply([1.0, 0.0, 0.0])
        normal *= float(obs["gate_sequence_direction"][self.index])
        return gate, center, normal

    def _path(self, pos: NDArray, target: NDArray, obs: dict) -> list[NDArray]:
        """Find a short collision-buffered polyline with a visibility graph and Dijkstra."""
        # Visibility graph in the horizontal plane. Unknown objects get larger margins.
        centers = []
        radii = []
        for p, seen in zip(obs["obstacles_pos"], obs["obstacles_visited"]):
            centers.append(p[:2])
            radii.append(0.19 if seen else 0.40)
        for gate_id, (p, q, seen) in enumerate(
            zip(obs["gates_pos"], obs["gates_quat"], obs["gates_visited"])
        ):
            lateral = Rotation.from_quat(q).apply([0.0, 1.0, 0.0])[:2]
            for offset in (
                (-0.28, 0.28)
                if seen and gate_id == int(obs["gate_sequence"][self.index])
                else (-0.28, 0.0, 0.28)
            ):
                centers.append(p[:2] + offset * lateral)
                radii.append(0.19 if seen else 0.36)
        centers, radii = np.asarray(centers), np.asarray(radii)
        # Allow escape from a newly observed inflated region, without shrinking other regions.
        radii = np.minimum(radii, np.linalg.norm(centers - pos[:2], axis=1) - 0.005)
        radii = np.maximum(radii, 0.01)
        if np.any(np.linalg.norm(centers - target[:2], axis=1) < radii - 1e-6):
            return []
        delta = target[:2] - pos[:2]
        t = np.clip((centers - pos[:2]) @ delta / max(delta @ delta, 1e-12), 0, 1)
        if np.all(np.linalg.norm(pos[:2] + t[:, None] * delta - centers, axis=1) >= radii - 1e-6):
            return [np.array(target, dtype=float)]
        points = [pos[:2], target[:2]]
        for center, radius in zip(centers, radii):
            for angle in np.linspace(0, 2 * np.pi, 10, endpoint=False):
                p = center + (radius + 0.07) * np.array([np.cos(angle), np.sin(angle)])
                if -2.3 < p[0] < 2.3 and -1.35 < p[1] < 1.35:
                    if np.all(np.linalg.norm(centers - p, axis=1) > radii):
                        points.append(p)
        points = np.asarray(points)
        edges = [[] for _ in points]
        for i in range(len(points)):
            delta = points[i + 1 :] - points[i]
            if not len(delta):
                continue
            length2 = np.sum(delta * delta, axis=1)
            t = np.clip(
                np.einsum("kd,jd->jk", centers - points[i], delta)
                / np.maximum(length2[:, None], 1e-12),
                0,
                1,
            )
            closest = points[i] + t[..., None] * delta[:, None, :]
            valid = np.all(np.linalg.norm(closest - centers, axis=-1) >= radii - 1e-6, axis=1)
            for j in np.flatnonzero(valid):
                k = i + 1 + int(j)
                cost = float(np.sqrt(length2[j]))
                edges[i].append((k, cost))
                edges[k].append((i, cost))
        queue = [(0.0, 0)]
        costs = {0: 0.0}
        parent = {}
        while queue:
            cost, node = heapq.heappop(queue)
            if node == 1:
                break
            if cost != costs[node]:
                continue
            for nxt, weight in edges[node]:
                nc = cost + weight
                if nc < costs.get(nxt, np.inf):
                    costs[nxt], parent[nxt] = nc, node
                    heapq.heappush(queue, (nc, nxt))
        if 1 not in parent:
            return []
        ids = [1]
        while ids[-1] != 0:
            ids.append(parent[ids[-1]])
        ids = ids[::-1][1:]
        result = []
        for i in ids:
            result.append(np.array([*points[i], target[2]]))
        return result

    def _approach(self, obs: dict):
        """Select a reachable observation point or a sensed gate-axis entry point."""
        gate, center, normal = self._geometry(obs)
        self.known = bool(obs["gates_visited"][gate])
        lateral = Rotation.from_quat(obs["gates_quat"][gate]).apply([0.0, 1.0, 0.0])
        if self.known:
            candidates = [center - d * normal for d in (0.24, 0.18, 0.14, 0.3, 0.4, 0.5)]
        else:
            candidates = [
                center - d * normal + side * lateral
                for d in (0.4, 0.5, 0.6)
                for side in (0.0, -0.25, 0.25, -0.4, 0.4)
            ]
        self.seen = (obs["gates_visited"].copy(), obs["obstacles_visited"].copy())
        self.route = []
        exits = [
            center + forward * normal + side * lateral
            for forward, side in (
                (0.24, 0.0),
                (0.22, 0.14),
                (0.22, -0.14),
                (0.18, 0.18),
                (0.18, -0.18),
            )
        ]
        self.exit_point = exits[0]
        for entry in candidates:
            if any(
                np.linalg.norm(entry[:2] - p[:2]) < (0.17 if seen else 0.41)
                for p, seen in zip(obs["obstacles_pos"], obs["obstacles_visited"])
            ):
                continue
            if self.known:
                clear_exit = None
                obstacles = np.asarray(obs["obstacles_pos"])[:, :2]
                for end in exits:
                    delta = end[:2] - entry[:2]
                    t = np.clip((obstacles - entry[:2]) @ delta / max(delta @ delta, 1e-9), 0, 1)
                    if np.all(
                        np.linalg.norm(entry[:2] + t[:, None] * delta - obstacles, axis=1) > 0.16
                    ):
                        clear_exit = end
                        break
                if clear_exit is None:
                    continue
                self.exit_point = clear_exit
            self.route = self._path(np.asarray(obs["pos"]), entry, obs)
            if self.route:
                break
        self.phase = "approach"
        self.blocked = not self.route
        self._set_target(self.route.pop(0) if self.route else obs["pos"])

    def compute_control(self, obs: dict, info: dict | None = None) -> NDArray:
        """Follow the approach/cross state machine and return a 16-component state command."""
        pos = np.asarray(obs["pos"])
        self.velocity = np.asarray(obs["vel"])
        if self.phase == "approach":
            if self.seen is not None and (
                np.any(obs["gates_visited"] != self.seen[0])
                or np.any(obs["obstacles_visited"] != self.seen[1])
            ):
                # Replan only from public observations, never from the simulator's hidden state.
                self._approach(obs)
        tolerance = (
            0.05 if self.phase == "cross" or (self.phase == "approach" and not self.route) else 0.10
        )
        done = np.linalg.norm(pos - self.goal) < tolerance and np.linalg.norm(self.velocity) < 0.3
        if self.phase == "cross":
            _, center, normal = self._geometry(obs)
            done = (
                np.dot(pos - center, normal) >= np.dot(self.goal - center, normal) - 0.02
                and np.linalg.norm(pos - self.goal) < 0.10
                and np.linalg.norm(self.velocity) < 0.3
            )
        if done:
            if self.phase == "takeoff":
                self._approach(obs)
            elif self.phase == "approach":
                if self.blocked:
                    pass
                elif self.route:
                    self._set_target(self.route.pop(0))
                elif not self.known:
                    self._approach(obs)
                else:
                    gate, center, normal = self._geometry(obs)
                    self.phase = "cross"
                    self._set_target(self.exit_point)
            elif self.phase == "cross":
                passed = int(obs["n_gates_passed"])
                if passed > self.index:
                    self.index = passed
                if self.index < len(obs["gate_sequence"]):
                    self._approach(obs)
        error = self.goal - pos
        distance = np.linalg.norm(error)
        self.last_setpoint = pos + error * min(1.0, 0.22 / max(distance, 1e-9))
        desired_velocity = error * min(1.8, 0.8 / max(distance, 1e-9))
        self.bias[2] = np.clip(self.bias[2] + self.dt * 0.5 * error[2], -0.18, 0.18)
        return np.concatenate(
            (
                self.last_setpoint + self.bias,
                desired_velocity - 0.5 * obs["vel"],
                np.zeros(3),
                [0.0, 0.0, 0.0, 1.0],
                np.zeros(3),
            )
        ).astype(np.float32)

    def step_callback(
        self,
        action: NDArray,
        obs: dict,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Let the race environment decide completion, collision, and timeout."""
        return False
