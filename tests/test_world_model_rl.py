import unittest
import importlib.machinery
import importlib.util
import sys
import torch

loader = importlib.machinery.SourceFileLoader('wmrl', 'src/world_model_rl.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
wmrl = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = wmrl
loader.exec_module(wmrl)
RLBridgeConfig = wmrl.RLBridgeConfig
TransitionDataset = wmrl.TransitionDataset
train_world_model = wmrl.train_world_model
rollout_policy = wmrl.rollout_policy


class TestWorldModelRL(unittest.TestCase):
    def setUp(self):
        self.cfg = RLBridgeConfig(state_dim=3, action_dim=2, epochs=1, batch_size=2)
        transitions = []
        for _ in range(4):
            s = torch.randn(3)
            a = torch.randint(0, 2, (1,)).item()
            ns = torch.randn(3)
            r = torch.randn(())  # float32 tensor
            transitions.append((s, a, ns, r))
        self.dataset = TransitionDataset(transitions)

    def test_training_and_rollout(self):
        model = train_world_model(self.cfg, self.dataset)
        self.assertIsInstance(model, torch.nn.Module)

        def policy(state: torch.Tensor) -> torch.Tensor:
            return torch.zeros((), dtype=torch.long)

        init_state = torch.zeros(3)
        states, rewards = rollout_policy(model, policy, init_state, steps=3)
        self.assertEqual(len(states), 3)
        self.assertEqual(len(rewards), 3)


if __name__ == "__main__":
    unittest.main()
