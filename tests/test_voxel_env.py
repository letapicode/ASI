import unittest
import importlib.machinery
import importlib.util
import sys
import torch

loader = importlib.machinery.SourceFileLoader('spe', 'src/self_play_env.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
spe = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = spe
loader.exec_module(spe)
VoxelEnv = spe.VoxelEnv
rollout_env = spe.rollout_env

class TestVoxelEnv(unittest.TestCase):
    def test_voxel_observation_shape(self):
        env = VoxelEnv((2,2,2))
        policy = lambda obs: torch.ones_like(obs)
        obs, rewards = rollout_env(env, policy, steps=1)
        self.assertEqual(obs[0].shape, torch.Size([2,2,2]))

if __name__ == '__main__':
    unittest.main()
