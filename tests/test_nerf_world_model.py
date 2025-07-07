import unittest
import importlib.machinery
import importlib.util
import sys
import torch
import types
from pathlib import Path

asi_pkg = types.ModuleType('asi')
sys.modules.setdefault('asi', asi_pkg)
src_pkg = types.ModuleType('src')
src_pkg.__path__ = [str(Path('src'))]
sys.modules.setdefault('src', src_pkg)
sys.modules.setdefault('psutil', types.SimpleNamespace())

# dynamically load modules
loader = importlib.machinery.SourceFileLoader('nerf_world_model', 'src/nerf_world_model.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
nerf = importlib.util.module_from_spec(spec)
sys.modules['nerf_world_model'] = nerf
loader.exec_module(nerf)
TinyNeRF = nerf.TinyNeRF
TinyNeRFCfg = nerf.TinyNeRFCfg
RayDataset = nerf.RayDataset
train_nerf = nerf.train_nerf

loader = importlib.machinery.SourceFileLoader('wmrl', 'src/world_model_rl.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
wmrl = importlib.util.module_from_spec(spec)
sys.modules['wmrl'] = wmrl
wmrl.__package__ = 'src'
loader.exec_module(wmrl)
RLBridgeConfig = wmrl.RLBridgeConfig
TransitionDataset = wmrl.TransitionDataset
train_world_model = wmrl.train_world_model
rollout_policy = wmrl.rollout_policy


class TestNeRFWorldModel(unittest.TestCase):
    def setUp(self):
        pose = torch.eye(4)
        img = torch.full((3, 2, 2), 0.5)
        self.dataset = RayDataset([(pose, img)])
        cfg = TinyNeRFCfg(hidden_dim=8, epochs=1, lr=1e-2)
        self.nerf = TinyNeRF(cfg)
        train_nerf(self.nerf, self.dataset, batch_size=2)

    def test_render_shape(self):
        frame = self.nerf.render(torch.eye(4), (2, 2))
        self.assertEqual(frame.shape, torch.Size([3, 2, 2]))

    def test_rollout_with_nerf(self):
        rl_cfg = RLBridgeConfig(state_dim=3, action_dim=1, epochs=1, batch_size=1)
        data = TransitionDataset([(torch.zeros(3), 0, torch.zeros(3), torch.tensor(0.0))])
        wm = train_world_model(rl_cfg, data)
        policy = lambda s: torch.tensor(0)
        states, rewards, frames = rollout_policy(
            wm, policy, torch.zeros(3), steps=1, nerf=self.nerf, nerf_views=[torch.eye(4)], img_hw=(2, 2)
        )
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].shape, torch.Size([3, 2, 2]))


if __name__ == '__main__':
    unittest.main()
