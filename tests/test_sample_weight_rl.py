import importlib.machinery
import importlib.util
import sys
import types
import unittest
import torch

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg

loader = importlib.machinery.SourceFileLoader('asi.adaptive_curriculum', 'src/adaptive_curriculum.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
adaptive_curriculum = importlib.util.module_from_spec(spec)
sys.modules['asi.adaptive_curriculum'] = adaptive_curriculum
loader.exec_module(adaptive_curriculum)
SampleWeightRL = adaptive_curriculum.SampleWeightRL

class TestSampleWeightRL(unittest.TestCase):
    def test_update_weights(self):
        rl = SampleWeightRL(2, lr=0.5)
        before = rl.weights().clone()
        rl.update(0, 1.0)
        rl.update(1, -1.0)
        after = rl.weights()
        self.assertGreater(after[0].item(), before[0].item())

if __name__ == '__main__':
    unittest.main()
