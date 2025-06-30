import unittest

from asi.self_play_env import SimpleGrid, SelfPlayAgent, rollout_env


class TestSelfPlayEnv(unittest.TestCase):
    def test_rollout(self):
        env = SimpleGrid(size=3)
        agent = SelfPlayAgent(env.action_space)
        obs, rewards = rollout_env(env, agent, steps=5)
        self.assertGreaterEqual(len(obs), 1)
        self.assertLessEqual(len(rewards), 5)


if __name__ == "__main__":
    unittest.main()
