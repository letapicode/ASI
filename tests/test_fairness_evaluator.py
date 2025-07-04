import unittest
import importlib.machinery
import importlib.util
import types
import sys

src_pkg = types.ModuleType('src')
src_pkg.__path__ = ['src']
src_pkg.__spec__ = importlib.machinery.ModuleSpec('src', None, is_package=True)
sys.modules['src'] = src_pkg
loader = importlib.machinery.SourceFileLoader('src.fairness_evaluator', 'src/fairness_evaluator.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
fe = importlib.util.module_from_spec(spec)
fe.__package__ = 'src'
sys.modules['src.fairness_evaluator'] = fe
loader.exec_module(fe)
FairnessEvaluator = fe.FairnessEvaluator

class TestFairnessEvaluator(unittest.TestCase):
    def test_metrics(self):
        stats = {
            'g1': {'tp': 10, 'fp': 5, 'fn': 5, 'tn': 80},
            'g2': {'tp': 20, 'fp': 10, 'fn': 0, 'tn': 70},
        }
        ev = FairnessEvaluator()
        res = ev.evaluate(stats)
        self.assertIn('demographic_parity', res)
        self.assertIn('equal_opportunity', res)
        self.assertGreaterEqual(res['demographic_parity'], 0)
        self.assertGreaterEqual(res['equal_opportunity'], 0)

if __name__ == '__main__':
    unittest.main()
