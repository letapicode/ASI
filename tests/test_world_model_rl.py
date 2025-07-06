import unittest
import importlib.machinery
import importlib.util
import types
import sys
import torch

pkg = types.ModuleType("src")
pkg.__path__ = ["src"]
pkg.__spec__ = importlib.machinery.ModuleSpec("src", None, is_package=True)
sys.modules["src"] = pkg

mods = ["self_play_env", "world_model_rl", "sim2real_adapter"]
for m in mods:
    loader = importlib.machinery.SourceFileLoader(f"src.{m}", f"src/{m}.py")
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "src"
    sys.modules[f"src.{m}"] = mod
    loader.exec_module(mod)

wmrl = sys.modules["src.world_model_rl"]
RLBridgeConfig = wmrl.RLBridgeConfig
TransitionDataset = wmrl.TransitionDataset
train_world_model = wmrl.train_world_model
rollout_policy = wmrl.rollout_policy
learn_env_params = sys.modules["src.sim2real_adapter"].learn_env_params
apply_correction = sys.modules["src.sim2real_adapter"].apply_correction
Sim2RealParams = sys.modules["src.sim2real_adapter"].Sim2RealParams


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

    def test_calibrated_training(self):
        logs = [
            (torch.zeros(3), torch.ones(3)),
            (torch.ones(3), torch.ones(3) * 2),
        ]
        model = train_world_model(
            self.cfg,
            self.dataset,
            calibration_traces=logs,
        )
        self.assertIsInstance(model, torch.nn.Module)


if __name__ == "__main__":
    unittest.main()
