import importlib.machinery
import importlib.util
import types
import dataclasses
import sys
import unittest
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - fallback stub
    import types
    torch = types.SimpleNamespace(
        tensor=lambda data, dtype=None: data,
        zeros=lambda s: [0.0] * (s[0] if isinstance(s, tuple) else s),
        ones=lambda *s: [1.0] * (s[0] if isinstance(s, tuple) else s),
        Tensor=list,
        nn=types.SimpleNamespace(Module=object, Linear=lambda *a, **k: object()),
    )
    sys.modules['torch'] = torch

sys.modules['psutil'] = types.SimpleNamespace()

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - fallback stub
    np = types.SimpleNamespace(
        zeros=lambda s, dtype=None: ([ [0.0]*s[1] for _ in range(s[0]) ] if isinstance(s, tuple) else [0.0]*s),
        ones=lambda s, dtype=None: ([ [1.0]*s[1] for _ in range(s[0]) ] if isinstance(s, tuple) else [1.0]*s),
        sin=lambda arr: [__import__('math').sin(x) for x in arr],
        linspace=lambda a, b, num, dtype=None: [a + (b - a) * i / (num - 1) for i in range(num)],
        pi=3.1415926,
        float32=float,
        ndarray=list,
    )
    sys.modules['numpy'] = np

loader_fb = importlib.machinery.SourceFileLoader('src.bci_feedback_trainer', 'src/bci_feedback_trainer.py')
spec_fb = importlib.util.spec_from_loader(loader_fb.name, loader_fb)
mod_fb = importlib.util.module_from_spec(spec_fb)
mod_fb.__package__ = 'src'
src_pkg = types.ModuleType('src')
src_pkg.__path__ = ['src']
src_pkg.__spec__ = importlib.machinery.ModuleSpec('src', None, is_package=True)
sys.modules['src'] = src_pkg
sys.modules['src.bci_feedback_trainer'] = mod_fb

# Minimal stub for world_model_rl to avoid heavy deps
mod_wm = types.ModuleType('src.world_model_rl')
@dataclasses.dataclass
class RLBridgeConfig:
    state_dim: int
    action_dim: int
    epochs: int = 1
    batch_size: int = 1
mod_wm.RLBridgeConfig = RLBridgeConfig
class TransitionDataset(list):
    pass
mod_wm.TransitionDataset = TransitionDataset
mod_wm.train_world_model = lambda *a, **k: torch.nn.Module()
sys.modules['src.world_model_rl'] = mod_wm

loader_fb.exec_module(mod_fb)
BCIFeedbackTrainer = mod_fb.BCIFeedbackTrainer
SecureFederatedLearner = __import__('src.secure_federated_learner', fromlist=['SecureFederatedLearner']).SecureFederatedLearner

cb_loader = importlib.machinery.SourceFileLoader('src.compute_budget_tracker', 'src/compute_budget_tracker.py')
cb_spec = importlib.util.spec_from_loader(cb_loader.name, cb_loader)
cbm = importlib.util.module_from_spec(cb_spec)
cbm.__package__ = 'src'
sys.modules['src.compute_budget_tracker'] = cbm
cb_loader.exec_module(cbm)

mod_wm = types.ModuleType('src.world_model_rl')
@dataclasses.dataclass
class RLBridgeConfig:
    state_dim: int
    action_dim: int
    epochs: int = 1
    batch_size: int = 1
mod_wm.RLBridgeConfig = RLBridgeConfig
class TransitionDataset(list):
    pass
mod_wm.TransitionDataset = TransitionDataset
mod_wm.train_world_model = lambda *a, **k: torch.nn.Module()
sys.modules['src.world_model_rl'] = mod_wm


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

    def test_discomfort_events(self):
        rl_cfg = RLBridgeConfig(state_dim=2, action_dim=2, epochs=1, batch_size=1)
        trainer = BCIFeedbackTrainer(rl_cfg)
        states = [torch.zeros(2)]
        actions = [0]
        next_states = [torch.ones(2)]
        t = np.linspace(0, 1, 32, dtype=np.float32)
        sig = np.sin(2 * np.pi * 40 * t)  # high-frequency to trigger discomfort
        trainer.train(states, actions, next_states, [sig])
        self.assertTrue(trainer.feedback_history[0])

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
