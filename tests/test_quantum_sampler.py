import unittest
import importlib.machinery
import importlib.util
import types
import sys
import numpy as np

src_pkg = types.ModuleType('src')
sys.modules['src'] = src_pkg

loader = importlib.machinery.SourceFileLoader('src.quantum_sampling', 'src/quantum_sampling.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
qs = importlib.util.module_from_spec(spec)
qs.__package__ = 'src'
sys.modules['src.quantum_sampling'] = qs
loader.exec_module(qs)
sample_actions_qae = qs.sample_actions_qae


class TestQuantumSampler(unittest.TestCase):
    def test_sample(self):
        idx = sample_actions_qae([1.0, 2.0])
        self.assertIn(idx, [0, 1])


if __name__ == '__main__':
    unittest.main()
