import unittest
import torch
import numpy as np
import importlib.util
import importlib.machinery
import types
import sys
from pathlib import Path

asi_pkg = types.ModuleType('asi')
sys.modules.setdefault('asi', asi_pkg)
src_pkg = types.ModuleType('src')
src_pkg.__path__ = [str(Path('src'))]
sys.modules.setdefault('src', src_pkg)

loader = importlib.machinery.SourceFileLoader('src.dp_memory', 'src/dp_memory.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
dp = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = dp
loader.exec_module(dp)
DifferentialPrivacyMemory = dp.DifferentialPrivacyMemory

class TestDifferentialPrivacyMemory(unittest.TestCase):
    def test_noise_and_retrieval(self):
        torch.manual_seed(0)
        np.random.seed(0)
        mem = DifferentialPrivacyMemory(dim=4, compressed_dim=2, capacity=10, dp_epsilon=0.5)
        data = torch.randn(2, 4)
        mem.add(data, metadata=['a', 'b'])
        stored = mem.store.search(mem.compressor.encoder(data[0]).detach().cpu().numpy(), k=1)[0]
        self.assertFalse(np.allclose(stored, mem.compressor.encoder(data[0]).detach().cpu().numpy()))
        out, meta = mem.search(data[0], k=1)
        self.assertEqual(meta[0], 'a')
        self.assertEqual(out.shape, (1, 4))

if __name__ == '__main__':
    unittest.main()
