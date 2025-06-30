import unittest
import torch
from torch import nn
from asi.lora_quant import LoRAQuantLinear, apply_quant_lora


class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)


class Nested(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 4)
        self.block = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))


class TestLoRAQuant(unittest.TestCase):
    def test_apply_and_forward(self):
        model = Dummy()
        apply_quant_lora(model, ["fc"], r=2)
        self.assertIsInstance(model.fc, LoRAQuantLinear)
        x = torch.randn(3, 4)
        y = model.fc(x)
        self.assertEqual(y.shape, (3, 2))
        model.fc.quantize()
        with torch.no_grad():
            y2 = model.fc(x)
        self.assertEqual(y2.shape, (3, 2))

    def test_nested_modules(self):
        model = Nested()
        apply_quant_lora(model, ["layer", "1"], r=2)
        self.assertIsInstance(model.layer, LoRAQuantLinear)
        self.assertIsInstance(model.block[1], LoRAQuantLinear)


if __name__ == "__main__":
    unittest.main()
