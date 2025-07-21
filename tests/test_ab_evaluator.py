import unittest
import json
import tempfile
import importlib.machinery
import importlib.util
import types
import sys
import numpy as np

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg
sys.modules['scripts'] = types.ModuleType('scripts')
pkg.__path__ = ['src']

torch = types.ModuleType('torch')
torch.randn = lambda *s: np.random.randn(*s).astype(np.float32)
torch.randint = lambda low, high, size: np.random.randint(low, high, size)
torch.cuda = types.SimpleNamespace(
    is_available=lambda: False,
    max_memory_allocated=lambda: 0,
    reset_peak_memory_stats=lambda: None,
)
sys.modules['torch'] = torch

PIL = types.ModuleType('PIL')
PIL.Image = object
sys.modules['PIL'] = PIL


def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = name.rpartition('.')[0]
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod

# Load minimal dependencies
_load('asi.fairness', 'src/fairness.py')
_load('asi.data_ingest', 'src/data_ingest.py')
_load('asi.cross_lingual_fairness', 'src/fairness.py')
_load('asi.emotion_detector', 'src/emotion_detector.py')
_load('asi.eval_harness', 'src/eval_harness.py')
ab_mod = _load('asi.ab_evaluator', 'src/ab_evaluator.py')

run_config = ab_mod.run_config
compare_results = ab_mod.compare_results


class TestABEvaluator(unittest.TestCase):
    def test_compare_output_contains_delta(self):
        with tempfile.NamedTemporaryFile('w+', suffix='.json') as a, tempfile.NamedTemporaryFile('w+', suffix='.json') as b:
            json.dump({'modules': ['cross_lingual_fairness']}, a)
            a.flush()
            json.dump({'modules': ['cross_lingual_fairness', 'emotion_detector']}, b)
            b.flush()
            res_a = run_config(a.name)
            res_b = run_config(b.name)
            out = compare_results(res_a, res_b)
            self.assertIn('delta=', out)


if __name__ == '__main__':
    unittest.main()

