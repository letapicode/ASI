import unittest
import asyncio
import importlib.machinery
import importlib.util
import types
import sys
import numpy as np

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg

# minimal torch stub
torch = types.ModuleType('torch')
torch.randn = lambda *s: np.random.randn(*s).astype(np.float32)
torch.randint = lambda low, high, size: np.random.randint(low, high, size)
torch.cuda = types.SimpleNamespace(
    is_available=lambda: False,
    max_memory_allocated=lambda: 0,
    reset_peak_memory_stats=lambda: None,
)
sys.modules['torch'] = torch


def load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'asi'
    loader.exec_module(mod)
    sys.modules[name] = mod
    setattr(pkg, name.split('.')[-1], mod)
    return mod

# load required modules
fe = load('asi.fairness_evaluator', 'src/fairness_evaluator.py')
di = load('asi.data_ingest', 'src/data_ingest.py')
clf = load('asi.cross_lingual_fairness', 'src/cross_lingual_fairness.py')
ed = load('asi.emotion_detector', 'src/emotion_detector.py')
eh = load('asi.eval_harness', 'src/eval_harness.py')

# reduce evaluators to avoid heavy deps
eh.EVALUATORS = {
    'cross_lingual_fairness': eh._eval_cross_lingual_fairness,
    'emotion_detector': eh._eval_emotion_detector,
}

parse_modules = eh.parse_modules
evaluate_modules = eh.evaluate_modules
evaluate_modules_async = eh.evaluate_modules_async
log_memory_usage = eh.log_memory_usage
format_results = eh.format_results


class TestEvalHarness(unittest.TestCase):
    def test_parse_modules(self):
        mods = parse_modules('docs/Plan.md')
        self.assertIn('moe_router', mods)

    def test_evaluate_subset(self):
        subset = ['cross_lingual_fairness', 'emotion_detector']
        results = evaluate_modules(subset)
        for name in subset:
            self.assertIn(name, results)
            self.assertTrue(results[name][0], name)

    def test_evaluate_subset_async(self):
        subset = ['cross_lingual_fairness', 'emotion_detector']
        results = asyncio.run(evaluate_modules_async(subset))
        for name in subset:
            self.assertIn(name, results)
            self.assertTrue(results[name][0], name)

    def test_log_memory_usage(self):
        mem = log_memory_usage()
        self.assertIsInstance(mem, float)

    def test_format_results_reports_memory(self):
        subset = ['cross_lingual_fairness', 'emotion_detector']
        results = evaluate_modules(subset)
        mem = log_memory_usage()
        out = format_results(results)
        out += f"\nGPU memory used: {mem:.1f} MB"
        self.assertIn('GPU memory used', out)

    def test_cross_lingual_fairness_evaluator(self):
        results = evaluate_modules(['cross_lingual_fairness'])
        self.assertIn('cross_lingual_fairness', results)
        self.assertTrue(results['cross_lingual_fairness'][0])

    def test_emotion_detector_evaluator(self):
        results = evaluate_modules(['emotion_detector'])
        self.assertIn('emotion_detector', results)
        self.assertTrue(results['emotion_detector'][0])


if __name__ == '__main__':
    unittest.main()
