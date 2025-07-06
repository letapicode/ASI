import unittest
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path
import torch

asi_pkg = types.ModuleType('asi')
sys.modules.setdefault('asi', asi_pkg)
src_pkg = types.ModuleType('src')
src_pkg.__path__ = [str(Path('src'))]
sys.modules.setdefault('src', src_pkg)
sys.modules.setdefault('psutil', types.SimpleNamespace())

wmrl = importlib.import_module('src.world_model_rl')
RLBridgeConfig = wmrl.RLBridgeConfig
TransitionDataset = wmrl.TransitionDataset
train_world_model = wmrl.train_world_model
rollout_policy = wmrl.rollout_policy

odewm = importlib.import_module('src.ode_world_model')
ODEWorldModelConfig = odewm.ODEWorldModelConfig
train_ode_world_model = odewm.train_ode_world_model
rollout_ode = odewm.rollout_policy


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

    def test_ode_training_and_rollout(self):
        ode_cfg = ODEWorldModelConfig(state_dim=3, action_dim=2, epochs=1, batch_size=2)
        model = train_ode_world_model(ode_cfg, self.dataset)
        self.assertIsInstance(model, torch.nn.Module)

        def policy(state: torch.Tensor) -> torch.Tensor:
            return torch.zeros((), dtype=torch.long)

        init_state = torch.zeros(3)
        states, rewards = rollout_ode(model, policy, init_state, steps=3)
        self.assertEqual(len(states), 3)
        self.assertEqual(len(rewards), 3)


if __name__ == "__main__":
    unittest.main()
