"""Integration: full-start racing rollouts using A's engine and B's reward/GAE."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import gymnasium
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from training.team_a.policy import from_environment_action, to_environment_action
from training.team_b.gae import generalized_advantages
from training.team_b.reward import RewardWeights, racing_reward
from training.team_contract import RolloutBatch

if TYPE_CHECKING:
    from training.team_a.engine import StagePPOEngine
    from training.team_b.curriculum import HardCaseSampler


def make_environment(config: dict, seed: int) -> JaxToNumpy:
    """Use original race conditions with rendering disabled, never edit config files."""
    return JaxToNumpy(
        gymnasium.make(
            config.env.id,
            freq=config.env.freq,
            sim_config=config.sim,
            sensor_range=config.env.sensor_range,
            control_mode=config.env.control_mode,
            track=config.env.track,
            disturbances=config.env.get("disturbances"),
            randomizations=config.env.get("randomizations"),
            seed=seed,
        )
    )


class RaceCollector:
    """Collect complete episodes before updates; no teleport or implicit autoreset.

    Complete-episode collection is the first integration implementation. B's GAE
    still supports sampling cuts, but this runner avoids mid-episode checkpoint
    ambiguity and keeps all stages' future returns in the same update batch.
    """

    def __init__(
        self,
        env: gymnasium.Env,
        engine: StagePPOEngine,
        sampler: HardCaseSampler,
        run_id: str,
        weights: RewardWeights | None = None,
    ) -> None:
        """Keep one real environment and explicit per-episode action history."""
        self.env, self.engine, self.sampler = env, engine, sampler
        self.run_id = run_id
        self.weights = weights or RewardWeights()

    def collect(self, minimum_steps: int) -> tuple[RolloutBatch, list[dict]]:
        """Sample on-policy episodes, then compute GAE without masking gate changes."""
        if minimum_steps < 1:
            raise ValueError("minimum_steps must be positive")
        names = ["x", "z", "u", "old_log_prob", "old_value", "gate_stage", "actor_update_mask"]
        data = {name: [] for name in names}
        rewards, next_values, terminated_flags, timeouts = [], [], [], []
        episodes = []
        thrust_low, thrust_high = (
            float(self.env.action_space.low[3]),
            float(self.env.action_space.high[3]),
        )
        while len(rewards) < minimum_steps:
            seed = int(self.sampler.sample(1)[0])
            obs, _ = self.env.reset(seed=seed)
            previous = np.zeros((1, 4), dtype=np.float32)
            total_reward, parts_sum = 0.0, {}
            stage_counts: dict[int, int] = {}
            max_sampling_ms = 0.0
            for index in range(1501):
                x = self.engine.encode_obs([obs], previous)
                stage = np.array([int(obs["n_gates_passed"])], dtype=np.int64)
                started = time.perf_counter()
                step = self.engine.act(x, stage)
                max_sampling_ms = max(max_sampling_ms, 1000 * (time.perf_counter() - started))
                action = to_environment_action(step.u, thrust_low, thrust_high)[0]
                if not self.env.action_space.contains(action):
                    raise ValueError("A action mapping exceeds the original action space")
                actual_u = from_environment_action(action[None], thrust_low, thrust_high)
                current, _, terminated, truncated, _ = self.env.step(action)
                terminal, timeout = bool(terminated), bool(truncated)
                reward, parts = racing_reward(
                    obs,
                    current,
                    actual_u[0],
                    previous[0],
                    terminated=terminal,
                    truncated=timeout,
                    weights=self.weights,
                )
                if terminal or timeout:
                    next_value = 0.0
                else:
                    next_x = self.engine.encode_obs([current], actual_u)
                    next_stage = np.array([int(current["n_gates_passed"])], dtype=np.int64)
                    next_value = float(self.engine.value(next_x, next_stage)[0])
                for name, value in {
                    "x": x[0],
                    "z": step.z[0],
                    "u": step.u[0],
                    "old_log_prob": step.log_prob[0],
                    "old_value": step.value[0],
                    "gate_stage": stage[0],
                    "actor_update_mask": step.actor_update_mask[0],
                }.items():
                    data[name].append(np.array(value, copy=True))
                rewards.append(reward)
                next_values.append(next_value)
                terminated_flags.append(terminal)
                timeouts.append(timeout)
                total_reward += reward
                for name, value in parts.items():
                    parts_sum[name] = parts_sum.get(name, 0.0) + float(value)
                stage_counts[int(stage[0])] = stage_counts.get(int(stage[0]), 0) + 1
                obs, previous = current, actual_u
                if terminal or timeout:
                    break
            else:
                raise AssertionError("Official episode exceeded its task deadline")
            success = int(obs["n_gates_passed"]) == len(obs["gate_sequence"])
            self.sampler.observe(seed, success, int(obs["n_gates_passed"]))
            episodes.append(
                {
                    "seed": seed,
                    "success": success,
                    "gate_pass_count": int(obs["n_gates_passed"]),
                    "control_steps": index + 1,
                    "official_time": index / 50,
                    "training_reward": total_reward,
                    "reward_components": parts_sum,
                    "stage_steps": stage_counts,
                    "terminated": terminal,
                    "task_timeout": timeout,
                    "max_training_sampling_ms": max_sampling_ms,
                }
            )
        values = np.asarray(data["old_value"], dtype=np.float32)[:, None]
        advantages, returns = generalized_advantages(
            np.asarray(rewards)[:, None],
            values,
            np.asarray(next_values)[:, None],
            np.asarray(terminated_flags)[:, None],
            np.asarray(timeouts)[:, None],
        )
        batch = RolloutBatch(
            x=np.asarray(data["x"], dtype=np.float32),
            z=np.asarray(data["z"], dtype=np.float32),
            u=np.asarray(data["u"], dtype=np.float32),
            old_log_prob=np.asarray(data["old_log_prob"], dtype=np.float32),
            old_value=values[:, 0],
            returns=returns[:, 0],
            advantages=advantages[:, 0],
            gate_stage=np.asarray(data["gate_stage"], dtype=np.int64),
            actor_update_mask=np.asarray(data["actor_update_mask"], dtype=np.bool_),
            run_id=self.run_id,
            profile=self.engine.settings.profile,
        )
        return batch, episodes
