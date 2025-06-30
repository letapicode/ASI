import unittest

from asi.self_play_env import GridWorld, RandomAgent, rollout_env


class TestSelfPlayEnv(unittest.TestCase):
    def test_rollout(self):
        env = GridWorld(size=3)
        agent = RandomAgent()
        log = rollout_env(env, agent, steps=5)
        self.assertGreater(len(log), 0)
        self.assertTrue(all(hasattr(t, 'state') for t in log))


if __name__ == '__main__':
    unittest.main()
