import unittest
import torch

from asi.robot_skill_transfer import transfer_skills


class TestRobotSkillTransfer(unittest.TestCase):
    def test_transfer(self):
        pretrain = [(
            torch.randint(0, 10, (1, 3)),
            torch.randn(1, 3, 4),
            torch.randn(1, 3, 4),
            torch.randint(0, 5, (1, 3)),
        )]
        finetune = pretrain
        model = transfer_skills(pretrain, finetune, action_dim=5, epochs=1)
        text, img, aud, _ = pretrain[0]
        out = model(text, img, aud)
        self.assertEqual(out.shape, (1, 3, 5))


if __name__ == '__main__':
    unittest.main()
