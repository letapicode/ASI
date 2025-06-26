import unittest
import torch

from asi.critic_rlhf import CriticScorer, CriticRLHFTrainer


class TestCriticRLHFTrainer(unittest.TestCase):
    def test_scorer_penalty(self):
        scorer = CriticScorer(["bad"])
        self.assertEqual(scorer.score("bad stuff"), -1.0)
        self.assertEqual(scorer.score("all good"), 1.0)

    def test_training_guides_policy(self):
        torch.manual_seed(0)
        model = torch.nn.Linear(1, 2, bias=False)
        for p in model.parameters():
            torch.nn.init.constant_(p, 0.0)
        trainer = CriticRLHFTrainer(model, ["safe", "bad"], CriticScorer(["bad"]), lr=0.1)
        x = torch.zeros(1, 1)
        for _ in range(200):
            trainer.train_step(x)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=-1).squeeze()
        self.assertGreater(probs[0].item(), 0.7)


if __name__ == "__main__":
    unittest.main()
