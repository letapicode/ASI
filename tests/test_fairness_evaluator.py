import unittest
import importlib.machinery
import importlib.util
import types
import sys

src_pkg = types.ModuleType('src')
src_pkg.__path__ = ['src']
src_pkg.__spec__ = importlib.machinery.ModuleSpec('src', None, is_package=True)
sys.modules['src'] = src_pkg
loader = importlib.machinery.SourceFileLoader('src.fairness', 'src/fairness.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
fe = importlib.util.module_from_spec(spec)
fe.__package__ = 'src'
sys.modules['src.fairness'] = fe
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

    def test_multimodal(self):
        stats = {
            'image': {
                'g1': {'tp': 1, 'fn': 1},
                'g2': {'tp': 2, 'fn': 0},
            },
            'audio': {
                'g1': {'tp': 1, 'fn': 0},
                'g2': {'tp': 1, 'fn': 1},
            },
        }
        ev = FairnessEvaluator()
        res = ev.evaluate_multimodal(stats, positive_label='tp')
        self.assertEqual(set(res.keys()), {'image', 'audio'})
        self.assertIn('demographic_parity', res['image'])
        self.assertIn('equal_opportunity', res['audio'])

if __name__ == '__main__':
    unittest.main()
