import os
import unittest
import torch
import importlib.machinery
import importlib.util
import types
import sys

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg

def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    sys.modules[name] = mod
    return mod

PrivacyBudgetManager = _load('asi.privacy_budget_manager', 'src/privacy_budget_manager.py').PrivacyBudgetManager
_load('asi.streaming_compression', 'src/streaming_compression.py')
wm = _load('asi.world_model_rl', 'src/world_model_rl.py')
RLBridgeConfig = wm.RLBridgeConfig
TransitionDataset = wm.TransitionDataset
train_world_model = wm.train_world_model
DifferentialPrivacyConfig = _load('asi.differential_privacy_optimizer', 'src/differential_privacy_optimizer.py').DifferentialPrivacyConfig


class TestPrivacyBudgetManager(unittest.TestCase):
    def test_budget_tracking(self):
        log = "budget.json"
        pbm = PrivacyBudgetManager(1.0, 0.1, log)
        cfg = RLBridgeConfig(state_dim=2, action_dim=1, epochs=1, batch_size=1)
        trans = [(torch.zeros(2), 0, torch.zeros(2), 0.0)]
        ds = TransitionDataset(trans)
        dp = DifferentialPrivacyConfig(lr=0.1, clip_norm=1.0, noise_std=0.1)
        train_world_model(cfg, ds, dp, pbm=pbm, run_id="run1")
        eps, _ = pbm.remaining("run1")
        os.remove(log)
        self.assertLess(eps, 1.0)


if __name__ == "__main__":
    unittest.main()
