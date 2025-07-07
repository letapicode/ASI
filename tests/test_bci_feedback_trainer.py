import importlib.machinery
import importlib.util
import sys
import unittest
import torch
import numpy as np

loader_fb = importlib.machinery.SourceFileLoader('bfit', 'src/bci_feedback_trainer.py')
spec_fb = importlib.util.spec_from_loader(loader_fb.name, loader_fb)
mod_fb = importlib.util.module_from_spec(spec_fb)
mod_fb.__package__ = 'src'
sys.modules['src.bci_feedback_trainer'] = mod_fb
loader_fb.exec_module(mod_fb)
BCIFeedbackTrainer = mod_fb.BCIFeedbackTrainer

loader_wm = importlib.machinery.SourceFileLoader('wm', 'src/world_model_rl.py')
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
        signals = [np.ones((2, 2)), np.zeros((2, 2))]
        model = trainer.train(states, actions, next_states, signals)
        self.assertIsInstance(model, torch.nn.Module)


if __name__ == '__main__':
    unittest.main()
