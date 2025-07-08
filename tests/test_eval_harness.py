import unittest
import asyncio
import importlib.machinery
import importlib.util
import types
import sys
try:
    import numpy as np
except Exception:  # pragma: no cover - fallback stub
    np = types.SimpleNamespace(
        random=types.SimpleNamespace(
            randn=lambda *s: [0.0 for _ in range(int(__import__('functools').reduce(lambda a,b:a*b,s,1)))] if s else [0.0],
            randint=lambda low, high, size=None: 0,
        ),
        array=lambda x, dtype=None: x,
        ndarray=list,
        zeros=lambda s: [0.0] * (s[0] if isinstance(s, tuple) else s),
        ones_like=lambda x: [1.0 for _ in range(len(x))],
        exp=lambda x: 1.0,
        log=lambda x: 0.0,
        float32=float,
    )
    sys.modules['numpy'] = np

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg
pkg.__path__ = ['src']
# minimal stubs for modules imported by eval_harness
class _Dash:
    def __init__(self):
        self.records = []

    def start(self, port=0):
        pass

    def record(self, *args):
        self.records.append(args)

    def aggregate(self):
        return {"pass_rate": 1.0, "flagged_examples": len(self.records)}

sys.modules['asi.alignment_dashboard'] = types.ModuleType('alignment_dashboard')
sys.modules['asi.alignment_dashboard'].AlignmentDashboard = _Dash
sys.modules['asi.deliberative_alignment'] = types.ModuleType('deliberative_alignment')
sys.modules['asi.deliberative_alignment'].DeliberativeAligner = lambda *_: None
sys.modules['asi.iter_align'] = types.ModuleType('iter_align')
sys.modules['asi.iter_align'].IterativeAligner = lambda *_: None
sys.modules['asi.critic_rlhf'] = types.ModuleType('critic_rlhf')
sys.modules['asi.critic_rlhf'].CriticScorer = lambda *a, **k: types.SimpleNamespace(score=lambda t: 0)
sys.modules['asi.critic_rlhf'].CriticRLHFTrainer = lambda *a, **k: None
sys.modules['requests'] = types.ModuleType('requests')
sys.modules['aiohttp'] = types.ModuleType('aiohttp')
data_ingest_stub = types.ModuleType('asi.data_ingest')
class CrossLingualTranslator:
    def translate(self, text, lang):
        if text.startswith("[" + lang):
            return text
        return f"[{lang}] {text}"
data_ingest_stub.CrossLingualTranslator = CrossLingualTranslator
sys.modules['asi.data_ingest'] = data_ingest_stub
tb = types.ModuleType('textblob')
tb.TextBlob = lambda t='': types.SimpleNamespace(sentiment=types.SimpleNamespace(polarity=0.0))
sys.modules['textblob'] = tb

# minimal torch stub
torch = types.ModuleType('torch')
torch.randn = lambda *s: np.random.randn(*s).astype(np.float32)
torch.randint = lambda low, high, size: np.random.randint(low, high, size)
torch.tensor = lambda data, dtype=None: np.array(data)
torch.Tensor = np.ndarray
torch.cuda = types.SimpleNamespace(
    is_available=lambda: False,
    max_memory_allocated=lambda: 0,
    reset_peak_memory_stats=lambda: None,
)
torch.float32 = np.float32
torch.Tensor = np.ndarray
class DummyModule:
    def __init__(self):
        self.bias = None
    def __call__(self, x):
        return np.zeros((1, 2))
    def parameters(self):
        return []
    def register_parameter(self, name, param):
        setattr(self, name, param)
torch.nn = types.SimpleNamespace(Module=object, Linear=lambda *a, **k: DummyModule())
torch.nn.Parameter = lambda x: x
torch.zeros = lambda *s: np.zeros(s)
torch.ones_like = lambda x: np.ones_like(x)
torch.softmax = lambda logits, dim=-1: np.exp(logits) / np.exp(logits).sum(axis=dim, keepdims=True)
torch.multinomial = lambda probs, num: np.array([0])
torch.log = lambda x: np.log(x)
torch.optim = types.SimpleNamespace(SGD=lambda params, lr=0.01: types.SimpleNamespace(step=lambda: None, zero_grad=lambda: None))
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
clf = load('asi.cross_lingual_fairness', 'src/cross_lingual_fairness.py')
ed = load('asi.emotion_detector', 'src/emotion_detector.py')
load('asi.deliberative_alignment', 'src/deliberative_alignment.py')
load('asi.iter_align', 'src/iter_align.py')
load('asi.critic_rlhf', 'src/critic_rlhf.py')
eh = load('asi.eval_harness', 'src/eval_harness.py')

# reduce evaluators to avoid heavy deps
eh.EVALUATORS = {
    'cross_lingual_fairness': lambda: (True, 'ok'),
    'emotion_detector': lambda: (True, 'ok'),
}

parse_modules = eh.parse_modules
evaluate_modules = eh.evaluate_modules
evaluate_modules_async = eh.evaluate_modules_async
log_memory_usage = eh.log_memory_usage
format_results = eh.format_results
AlignmentDashboard = eh.AlignmentDashboard


class TestEvalHarness(unittest.TestCase):
    def test_parse_modules(self):
        mods = parse_modules('docs/Plan.md')
        self.assertIn('moe_router', mods)

    def test_evaluate_subset(self):
        subset = ['cross_lingual_fairness', 'emotion_detector']
        dash = AlignmentDashboard()
        results = evaluate_modules(subset, dash)
        stats = dash.aggregate()
        self.assertIn('pass_rate', stats)
        for name in subset:
            self.assertIn(name, results)
            self.assertTrue(results[name][0], name)

    def test_evaluate_subset_async(self):
        subset = ['cross_lingual_fairness', 'emotion_detector']
        dash = AlignmentDashboard()
        results = asyncio.run(evaluate_modules_async(subset, dash))
        stats = dash.aggregate()
        self.assertIn('flagged_examples', stats)
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
        dash = AlignmentDashboard()
        results = evaluate_modules(['cross_lingual_fairness'], dash)
        self.assertIn('cross_lingual_fairness', results)
        self.assertTrue(results['cross_lingual_fairness'][0])

    def test_emotion_detector_evaluator(self):
        dash = AlignmentDashboard()
        results = evaluate_modules(['emotion_detector'], dash)
        self.assertIn('emotion_detector', results)
        self.assertTrue(results['emotion_detector'][0])


if __name__ == '__main__':
    unittest.main()
