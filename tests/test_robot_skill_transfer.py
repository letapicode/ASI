import unittest
import importlib.machinery
import importlib.util
import sys
import torch

loader = importlib.machinery.SourceFileLoader('rst', 'src/robot_skill_transfer.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
rst = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = rst
loader.exec_module(rst)
SkillTransferConfig = rst.SkillTransferConfig
VideoPolicyDataset = rst.VideoPolicyDataset
transfer_skills = rst.transfer_skills


class TestRobotSkillTransfer(unittest.TestCase):
    def test_dataset_length_mismatch(self):
        frames = [torch.randn(3, 8, 8)] * 2
        actions = [0]
        with self.assertRaises(ValueError):
            VideoPolicyDataset(frames, actions)

    def test_training(self):
        frames = [torch.randn(3, 8, 8) for _ in range(3)]
        actions = [0, 1, 0]
        dataset = VideoPolicyDataset(frames, actions)
        cfg = SkillTransferConfig(img_channels=3, action_dim=2, epochs=1, batch_size=2)
        model = transfer_skills(cfg, dataset)
        self.assertIsInstance(model, torch.nn.Module)


if __name__ == "__main__":
    unittest.main()
