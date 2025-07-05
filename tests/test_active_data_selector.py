import importlib.machinery
import importlib.util
import sys
import types
import unittest
import numpy as np
import tempfile
from pathlib import Path

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg

loader_bias = importlib.machinery.SourceFileLoader(
    'asi.dataset_bias_detector', 'src/dataset_bias_detector.py'
)
spec_bias = importlib.util.spec_from_loader(loader_bias.name, loader_bias)
bias_mod = importlib.util.module_from_spec(spec_bias)
sys.modules['asi.dataset_bias_detector'] = bias_mod
loader_bias.exec_module(bias_mod)

loader = importlib.machinery.SourceFileLoader('asi.data_ingest', 'src/data_ingest.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
data_ingest = importlib.util.module_from_spec(spec)
sys.modules['asi.data_ingest'] = data_ingest
loader.exec_module(data_ingest)
ActiveDataSelector = data_ingest.ActiveDataSelector

class TestActiveDataSelector(unittest.TestCase):
    def test_select_weights(self):
        selector = ActiveDataSelector(threshold=1.0)
        with tempfile.TemporaryDirectory() as d:
            p1 = Path(d) / 'a.txt'
            p2 = Path(d) / 'b.txt'
            p1.write_text('alpha beta gamma delta')
            p2.write_text('hello hello hello hello')
            triples = [(p1, 'i', 'a'), (p2, 'i2', 'a2')]
            probs = [np.array([0.5, 0.5]), np.array([0.5, 0.5])]
            weights = selector.select(triples, probs)
            self.assertEqual(len(weights), 2)
            self.assertGreater(weights[0][1], weights[1][1])

if __name__ == '__main__':
    unittest.main()
