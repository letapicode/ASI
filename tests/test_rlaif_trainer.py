import unittest
import importlib.machinery
import importlib.util
import sys
import types
import numpy as np

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg

def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = name.rpartition('.')[0]
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod

_load('asi.critic_rlhf', 'src/critic_rlhf.py')
rlaif_mod = _load('asi.rlaif_trainer', 'src/rlaif_trainer.py')
RLAIFTrainer = rlaif_mod.RLAIFTrainer
SyntheticCritic = rlaif_mod.SyntheticCritic


class TestRLAIFTrainer(unittest.TestCase):
    def test_critic_scores(self):
        model = {"weight": np.array([[0.0], [1.0]]), "bias": np.zeros(2)}
        tok = lambda t: np.array([1.0]) if "good" in t else np.array([0.0])
        critic = SyntheticCritic(model, tok)
        self.assertEqual(critic.score("good"), 1.0)
        self.assertEqual(critic.score("bad"), -1.0)

    def test_training_guides_policy(self):
        np.random.seed(0)
        policy = {"weight": np.zeros((2, 1)), "bias": np.zeros(2)}
        critic_model = {"weight": np.array([[0.0], [1.0]]), "bias": np.zeros(2)}
        tok = lambda t: np.array([1.0]) if "good" in t else np.array([0.0])

        trainer = RLAIFTrainer(policy, ["good", "bad"], critic_model, tok, lr=0.1)
        x = np.zeros((1, 1))
        for _ in range(200):
            trainer.train_step(x)
        logits = x @ policy["weight"].T + policy["bias"]
        probs = np.exp(logits - np.max(logits))
        probs = probs / probs.sum()
        self.assertGreater(probs[0, 0], 0.7)


if __name__ == "__main__":
    unittest.main()
