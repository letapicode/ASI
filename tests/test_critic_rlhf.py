import unittest
from src.critic_rlhf import CriticRLHF


class TestCriticRLHF(unittest.TestCase):
    def test_update_combines_scores(self):
        trainer = CriticRLHF(actions=["a"], critic_weight=0.5, lr=1.0, epsilon=0.0)
        trainer.update("a", human_score=0.0, critic_score=1.0)
        self.assertAlmostEqual(trainer.values["a"], 0.5)

    def test_select_action_prefers_high_value(self):
        trainer = CriticRLHF(actions=["x", "y"], critic_weight=0.0, lr=1.0, epsilon=0.0)
        trainer.update("y", human_score=1.0, critic_score=0.0)
        action = trainer.select_action()
        self.assertEqual(action, "y")


if __name__ == "__main__":
    unittest.main()
