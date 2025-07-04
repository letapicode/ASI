import unittest
import torch
import importlib
import sys
import types
from pathlib import Path

asi_pkg = types.ModuleType('asi')
sys.modules.setdefault('asi', asi_pkg)
src_pkg = types.ModuleType('src')
src_pkg.__path__ = [str(Path('src'))]
sys.modules.setdefault('src', src_pkg)

RLBridgeConfig = importlib.import_module('src.world_model_rl').RLBridgeConfig
WorldModel = importlib.import_module('src.world_model_rl').WorldModel
WorldModelDebugger = importlib.import_module('src.world_model_debugger').WorldModelDebugger


class TestWorldModelDebugger(unittest.TestCase):
    def test_patch_trigger(self):
        cfg = RLBridgeConfig(state_dim=2, action_dim=2, hidden_dim=4, lr=1e-3, batch_size=2, epochs=1)
        model = WorldModel(cfg)
        debugger = WorldModelDebugger(model, threshold=0.1)
        states = torch.zeros(4, 2)
        actions = torch.zeros(4, dtype=torch.long)
        targets = torch.ones(4, 2)
        before = debugger.check(states, actions, targets)
        after = debugger.check(states, actions, targets)
        self.assertLessEqual(after, before)


if __name__ == '__main__':
    unittest.main()
