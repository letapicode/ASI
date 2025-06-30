import unittest
import torch

from asi.self_play_env import SimpleEnv, rollout_env


class TestSelfPlayEnv(unittest.TestCase):
    def test_step_and_reset(self):
        env = SimpleEnv(state_dim=3)
        obs = env.reset()
        self.assertTrue(torch.allclose(obs, torch.zeros(3)))
        step = env.step(torch.ones(3))
        self.assertEqual(step.observation.shape, (3,))
        self.assertIsInstance(step.reward, float)

    def test_rollout_env(self):
        env = SimpleEnv(state_dim=2)
        policy = lambda obs: torch.zeros_like(obs)
        obs_list, rewards = rollout_env(env, policy, steps=5)
        self.assertEqual(len(obs_list), 1)
        self.assertEqual(len(rewards), 1)
        self.assertTrue(env.state.norm() < 1e-6)


if __name__ == "__main__":
    unittest.main()
