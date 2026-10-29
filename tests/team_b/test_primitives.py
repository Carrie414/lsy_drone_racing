"""B1/B2 standalone contract tests; no invented actor or simulated PPO training."""

import unittest

import numpy as np

from training.team_b.curriculum import HardCaseSampler
from training.team_b.gae import generalized_advantages
from training.team_b.reward import racing_reward


def obs(stage=0, pos=None):
    return dict(
        n_gates_passed=stage,
        gate_sequence=np.array([0, 1, 0]),
        gate_sequence_direction=np.array([1, 1, -1]),
        gates_pos=np.array([[1.0, 0, 1], [2, 0, 1]]),
        pos=np.array(pos if pos is not None else [0.0, 0, 1]),
    )


class RewardTests(unittest.TestCase):
    def test_gate_and_finish_from_official_counter_only(self):
        _, p = racing_reward(obs(2), obs(3), np.zeros(4), np.zeros(4), terminated=True)
        self.assertEqual(p["gate"], 5)
        self.assertEqual(p["finish"], 20)
        self.assertEqual(p["failure"], 0)
        _, p = racing_reward(obs(3), obs(3), np.zeros(4), np.zeros(4), terminated=True)
        self.assertEqual(p["finish"], 0)

    def test_crossing_without_official_pass_gets_no_gate_reward(self):
        _, p = racing_reward(obs(pos=[0.9, 0, 1]), obs(pos=[1.1, 0, 1]), np.zeros(4), np.zeros(4))
        self.assertEqual(p["gate"], 0)

    def test_gate_reveal_does_not_create_progress(self):
        a = obs()
        b = obs()
        b["gates_pos"][0] = [0, 0, 1]
        _, p = racing_reward(a, b, np.zeros(4), np.zeros(4))
        self.assertEqual(p["progress"], 0)

    def test_target_change_does_not_compare_unrelated_targets(self):
        _, p = racing_reward(obs(0, [0.9, 0, 1]), obs(1, [1, 0, 1]), np.zeros(4), np.zeros(4))
        self.assertAlmostEqual(p["progress"], 0.1)

    def test_failure_masks_potential_from_terminal_warp(self):
        _, p = racing_reward(
            obs(), obs(pos=[0, 0, -100]), np.zeros(4), np.zeros(4), terminated=True
        )
        self.assertEqual(p["progress"], 0)
        self.assertEqual(p["failure"], -5)

    def test_autoreset_must_supply_final_observation(self):
        with self.assertRaises(ValueError):
            racing_reward(obs(2), obs(0), np.zeros(4), np.zeros(4))


class GAETests(unittest.TestCase):
    def test_stage_change_not_a_terminal(self):
        a, r = generalized_advantages(
            [[1], [2]], [[0], [0]], [[0], [0]], [[0], [1]], [[0], [0]], gamma=1, gae_lambda=1
        )
        np.testing.assert_allclose(a[:, 0], [3, 2])
        np.testing.assert_allclose(a, r)

    def test_sampling_cut_bootstraps_true_final_value(self):
        a, _ = generalized_advantages([[1]], [[0]], [[5]], [[0]], [[0]], gamma=0.9)
        self.assertAlmostEqual(float(a[0, 0]), 5.5)

    def test_task_timeout_does_not_bootstrap(self):
        a, _ = generalized_advantages([[1]], [[0]], [[100]], [[0]], [[1]])
        self.assertEqual(a[0, 0], 1)

    def test_no_cross_episode_or_cross_environment_leak(self):
        a, _ = generalized_advantages(
            [[1, 1], [99, 2]],
            [[0, 0], [0, 0]],
            [[100, 0], [0, 0]],
            [[1, 0], [1, 1]],
            [[0, 0], [0, 0]],
            gamma=1,
            gae_lambda=1,
        )
        np.testing.assert_allclose(a, [[1, 3], [99, 2]])


class CurriculumTests(unittest.TestCase):
    def test_reserved_seeds_rejected(self):
        with self.assertRaises(ValueError):
            HardCaseSampler([1, 2], forbidden_seeds=[2])

    def test_normal_pool_is_preserved_and_weights_bounded(self):
        c = HardCaseSampler(range(10))
        c.observe(0, False, 0)
        p = c.probabilities()
        self.assertAlmostEqual(p.sum(), 1)
        self.assertTrue((p >= 0.05).all())
        self.assertTrue(p.max() / p.min() < 3)
        self.assertTrue(set(c.sample(200)) <= set(range(10)))

    def test_sampler_restore_reproduces_next_draw(self):
        a = HardCaseSampler([1, 2, 3])
        a.observe(2, False, 2)
        state = a.state_dict()
        expected = a.sample(30)
        b = HardCaseSampler([1, 2, 3])
        b.load_state_dict(state)
        np.testing.assert_array_equal(b.sample(30), expected)


if __name__ == "__main__":
    unittest.main()
