import unittest
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


def load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'asi'
    sys.modules[name] = mod
    loader.exec_module(mod)
    setattr(pkg, name.split('.')[-1], mod)
    return mod

rh = load('asi.reasoning_history', 'src/reasoning_history.py')


class TestReasoningHistoryLogger(unittest.TestCase):
    def test_log_debate(self):
        logger = rh.ReasoningHistoryLogger()
        transcript = [('Q', 'hi'), ('A', 'hello')]
        logger.log_debate(transcript, 'ok')
        hist = logger.get_history()
        self.assertEqual(hist[0][1]['verdict'], 'ok')
        self.assertEqual(len(hist[0][1]['transcript']), 2)


if __name__ == '__main__':
    unittest.main()
