import unittest
import torch
from torch import nn
from asi.gradient_patch_editor import GradientPatchEditor

class TestGradientPatchEditor(unittest.TestCase):
    def test_patch(self):
        model = nn.Linear(2, 1)
        editor = GradientPatchEditor(model)
        x = torch.zeros(4, 2)
        y = torch.ones(4, 1)
        loss = editor.patch(x, y, nn.MSELoss())
        self.assertIsInstance(loss, float)

if __name__ == '__main__':
    unittest.main()
