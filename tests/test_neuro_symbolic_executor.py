import unittest
import importlib.machinery
import importlib.util
import types
import sys
import torch

# prepare asi package and load world_model_rl first
pkg = types.ModuleType('asi')
sys.modules.setdefault('asi', pkg)
loader_wm = importlib.machinery.SourceFileLoader('asi.world_model_rl', 'src/world_model_rl.py')
spec_wm = importlib.util.spec_from_loader(loader_wm.name, loader_wm)
wm = importlib.util.module_from_spec(spec_wm)
sys.modules[loader_wm.name] = wm
loader_wm.exec_module(wm)
WorldModel = wm.WorldModel
RLBridgeConfig = wm.RLBridgeConfig

# load NeuroSymbolicExecutor which imports world_model_rl relatively
loader = importlib.machinery.SourceFileLoader('asi.neuro_symbolic_executor', 'src/neuro_symbolic_executor.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
nse = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = nse
loader.exec_module(nse)
NeuroSymbolicExecutor = nse.NeuroSymbolicExecutor
ConstraintViolation = nse.ConstraintViolation


class TestNeuroSymbolicExecutor(unittest.TestCase):
    def setUp(self):
        cfg = RLBridgeConfig(state_dim=2, action_dim=2)
        self.model = WorldModel(cfg)

    def test_rollout_with_violation(self):
        def policy(state: torch.Tensor) -> torch.Tensor:
            return torch.zeros((), dtype=torch.long)

        # impossible constraint -> always fails
        def impossible(state, action, next_state):
            return False

        executor = NeuroSymbolicExecutor(self.model, [("fail", impossible)])
        states, rewards, violations = executor.rollout(policy, torch.zeros(2), steps=3)
        self.assertEqual(len(states), 3)
        self.assertEqual(len(rewards), 3)
        self.assertEqual(len(violations), 3)
        self.assertIsInstance(violations[0], ConstraintViolation)
        self.assertEqual(violations[0].message, "fail")

    def test_rollout_no_violation(self):
        def policy(state: torch.Tensor) -> torch.Tensor:
            return torch.zeros((), dtype=torch.long)

        def always_ok(state, action, next_state):
            return True

        executor = NeuroSymbolicExecutor(self.model, [("ok", always_ok)])
        _, _, violations = executor.rollout(policy, torch.zeros(2), steps=2)
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
