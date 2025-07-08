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

# minimal torch stub
torch = types.ModuleType('torch')
torch.randn = lambda *s: np.random.randn(*s).astype(np.float32)
torch.randint = lambda low, high, size: np.random.randint(low, high, size)
torch.Tensor = np.ndarray
torch.float32 = np.float32
torch.nn = types.SimpleNamespace(Module=object, Linear=lambda *a, **k: object())
torch.cuda = types.SimpleNamespace(
    is_available=lambda: False,
    max_memory_allocated=lambda: 0,
    reset_peak_memory_stats=lambda: None,
)
sys.modules['torch'] = torch
sys.modules['requests'] = types.ModuleType('requests')
sys.modules['PIL'] = types.ModuleType('PIL')
sys.modules['PIL.Image'] = types.ModuleType('Image')

# stub cryptography dependency
crypto = types.ModuleType('cryptography')
haz = types.ModuleType('cryptography.hazmat')
prim = types.ModuleType('cryptography.hazmat.primitives')
ci = types.ModuleType('cryptography.hazmat.primitives.ciphers')
aead = types.ModuleType('cryptography.hazmat.primitives.ciphers.aead')
class AESGCM: ...
aead.AESGCM = AESGCM
ci.aead = aead
prim.ciphers = ci
haz.primitives = prim
crypto.hazmat = haz
sys.modules['cryptography'] = crypto
sys.modules['cryptography.hazmat'] = haz
sys.modules['cryptography.hazmat.primitives'] = prim
sys.modules['cryptography.hazmat.primitives.ciphers'] = ci
sys.modules['cryptography.hazmat.primitives.ciphers.aead'] = aead


def load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'asi'
    sys.modules[name] = mod
    loader.exec_module(mod)
    setattr(pkg, name.split('.')[-1], mod)
    return mod

# minimal hierarchical memory stub
mem_mod = types.ModuleType('hierarchical_memory')
class HierarchicalMemory:
    def __init__(self, dim, compressed_dim, capacity):
        self.dim = dim
    def add(self, vec):
        pass
mem_mod.HierarchicalMemory = HierarchicalMemory
sys.modules['asi.hierarchical_memory'] = mem_mod

load('asi.adaptive_planner', 'src/adaptive_planner.py')
load('asi.reasoning_history', 'src/reasoning_history.py')
sd = load('asi.socratic_debate', 'src/socratic_debate.py')


class TestSocraticDebate(unittest.TestCase):
    def test_run_debate(self):
        from asi.adaptive_planner import AdaptivePlanner
        from asi.hierarchical_memory import HierarchicalMemory
        a = sd.DebateAgent('A', AdaptivePlanner(lambda s: 1.0, actions=['replace','Why']), HierarchicalMemory(4,2,5))
        b = sd.DebateAgent('B', AdaptivePlanner(lambda s: 1.0, actions=['replace','Why']), a.memory)
        debate = sd.SocraticDebate(a, b)
        transcript, verdict = debate.run_debate('replace foo', rounds=1)
        self.assertEqual(len(transcript), 3)
        self.assertIsInstance(verdict, str)
        self.assertTrue(debate.logger.get_history())


if __name__ == '__main__':
    unittest.main()
