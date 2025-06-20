import unittest

from src.meta_rl_refactor import MetaRLRefactorAgent


class TestMetaRLRefactorAgent(unittest.TestCase):
    def test_update_and_select(self):
        agent = MetaRLRefactorAgent(epsilon=0.0)
        s1 = "s1"
        s2 = "s2"
        agent.update(s1, "replace", 1.0, s2)
        self.assertIn((s1, "replace"), agent.q)
        self.assertGreater(agent.q[(s1, "replace")], 0.0)
        action = agent.select_action(s1)
        self.assertEqual(action, "replace")


if __name__ == "__main__":
    unittest.main()
