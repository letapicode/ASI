import unittest
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path
import torch

pkg = types.ModuleType('asi')
sys.modules.setdefault('asi', pkg)
src_pkg = types.ModuleType('src')
src_pkg.__path__ = [str(Path('src'))]
sys.modules.setdefault('src', src_pkg)

# load modules
loader = importlib.machinery.SourceFileLoader('nerf', 'src/nerf_world_model.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
nerf = importlib.util.module_from_spec(spec)
sys.modules['src.nerf_world_model'] = nerf
loader.exec_module(nerf)

loader = importlib.machinery.SourceFileLoader('wmrl', 'src/world_model_rl.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
wmrl = importlib.util.module_from_spec(spec)
sys.modules['src.world_model_rl'] = wmrl
loader.exec_module(wmrl)

NeRFConfig = nerf.NeRFConfig
MultiViewDataset = nerf.MultiViewDataset
train_nerf = nerf.train_nerf
render_views = nerf.render_views
psnr = nerf.psnr

RLBridgeConfig = wmrl.RLBridgeConfig
TransitionDataset = wmrl.TransitionDataset
train_world_model = wmrl.train_world_model
rollout_policy = wmrl.rollout_policy


class TestNeRFWorldModel(unittest.TestCase):
    def setUp(self):
        color = torch.tensor([0.5, 0.1, 0.2])
        img = color.view(3, 1, 1).expand(-1, 2, 2)
        poses = [torch.eye(4), torch.eye(4)]
        intr = torch.eye(3)
        self.dataset = MultiViewDataset([img, img], poses, intr)
        cfg = NeRFConfig(epochs=2, batch_size=4, hidden_dim=8, num_samples=2)
        self.nerf = train_nerf(cfg, self.dataset)
        self.intr = intr
        self.image_size = (2, 2)

    def test_reconstruction(self):
        rays_o, rays_d, target = nerf.image_to_rays(
            self.dataset.data[0][0].new_zeros(3, 2, 2), torch.eye(4), self.intr
        )
        pred = self.nerf.render(rays_o, rays_d)
        val = psnr(pred, target)
        self.assertGreater(val.item(), 10.0)

    def test_rollout_render(self):
        cfg = RLBridgeConfig(state_dim=3, action_dim=2, epochs=1, batch_size=2)
        trans = []
        for _ in range(2):
            s = torch.zeros(3)
            trans.append((s, 0, s, 0.0))
        wm = train_world_model(cfg, TransitionDataset(trans))

        def policy(state: torch.Tensor) -> torch.Tensor:
            return torch.zeros((), dtype=torch.long)

        frames, rewards = rollout_policy(
            wm,
            policy,
            torch.zeros(3),
            steps=1,
            nerf=self.nerf,
            views=[torch.eye(4)],
            intrinsics=self.intr,
            image_size=self.image_size,
        )
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].shape, (3, 2, 2))
        self.assertEqual(len(rewards), 1)


if __name__ == "__main__":
    unittest.main()
