import importlib.machinery
import importlib.util
import types
import sys
import unittest
import torch
import numpy as np

loader_fb = importlib.machinery.SourceFileLoader('src.bci_feedback_trainer', 'src/bci_feedback_trainer.py')
spec_fb = importlib.util.spec_from_loader(loader_fb.name, loader_fb)
mod_fb = importlib.util.module_from_spec(spec_fb)
mod_fb.__package__ = 'src'
src_pkg = types.ModuleType('src')
src_pkg.__path__ = ['src']
src_pkg.__spec__ = importlib.machinery.ModuleSpec('src', None, is_package=True)
sys.modules['src'] = src_pkg
sys.modules['src.bci_feedback_trainer'] = mod_fb
loader_fb.exec_module(mod_fb)
BCIFeedbackTrainer = mod_fb.BCIFeedbackTrainer
SecureFederatedLearner = __import__('src.secure_federated_learner', fromlist=['SecureFederatedLearner']).SecureFederatedLearner

cb_loader = importlib.machinery.SourceFileLoader('src.compute_budget_tracker', 'src/compute_budget_tracker.py')
cb_spec = importlib.util.spec_from_loader(cb_loader.name, cb_loader)
cbm = importlib.util.module_from_spec(cb_spec)
cbm.__package__ = 'src'
sys.modules['src.compute_budget_tracker'] = cbm
cb_loader.exec_module(cbm)

loader_wm = importlib.machinery.SourceFileLoader('src.world_model_rl', 'src/world_model_rl.py')
spec_wm = importlib.util.spec_from_loader(loader_wm.name, loader_wm)
mod_wm = importlib.util.module_from_spec(spec_wm)
mod_wm.__package__ = 'src'
sys.modules['src.world_model_rl'] = mod_wm
loader_wm.exec_module(mod_wm)
RLBridgeConfig = mod_wm.RLBridgeConfig


class TestBCIFeedbackTrainer(unittest.TestCase):
    def test_training(self):
        rl_cfg = RLBridgeConfig(state_dim=2, action_dim=2, epochs=1, batch_size=2)
        trainer = BCIFeedbackTrainer(rl_cfg)
        states = [torch.zeros(2), torch.zeros(2)]
        actions = [0, 1]
        next_states = [torch.ones(2), torch.ones(2)]
        signals = [np.ones((2, 2), dtype=np.float32), np.zeros((2, 2), dtype=np.float32)]
        model = trainer.train(states, actions, next_states, signals)
        self.assertIsInstance(model, torch.nn.Module)

    def test_federated_signals(self):
        rl_cfg = RLBridgeConfig(state_dim=2, action_dim=2, epochs=1, batch_size=2)
        trainer = BCIFeedbackTrainer(rl_cfg)
        states = [torch.zeros(2), torch.zeros(2)]
        actions = [0, 1]
        next_states = [torch.ones(2), torch.ones(2)]
        s1 = [np.ones(4, dtype=np.float32), np.zeros(4, dtype=np.float32)]
        s2 = [np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32)]
        learner = SecureFederatedLearner()
        model = trainer.train(
            states,
            actions,
            next_states,
            None,
            signals_nodes=[s1, s2],
            learner=learner,
        )
        self.assertIsInstance(model, torch.nn.Module)


if __name__ == '__main__':
    unittest.main()
