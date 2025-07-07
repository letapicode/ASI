import importlib.machinery
import importlib.util
import types
import sys
import unittest
import torch

pkg = types.ModuleType('asi')
pkg.__path__ = ['src']
sys.modules['asi'] = pkg

crypto = types.ModuleType('cryptography')
haz = types.ModuleType('cryptography.hazmat')
prim = types.ModuleType('cryptography.hazmat.primitives')
ci = types.ModuleType('cryptography.hazmat.primitives.ciphers')
aead = types.ModuleType('cryptography.hazmat.primitives.ciphers.aead')
class AESGCM:
    pass
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
sys.modules['requests'] = types.ModuleType('requests')
psutil_stub = types.SimpleNamespace(
    cpu_percent=lambda interval=None: 0.0,
    virtual_memory=lambda: types.SimpleNamespace(percent=0.0),
    net_io_counters=lambda: types.SimpleNamespace(bytes_sent=0, bytes_recv=0),
)
sys.modules['psutil'] = psutil_stub
pil_mod = types.ModuleType('PIL')
pil_mod.Image = object
sys.modules['PIL'] = pil_mod


def load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    loader.exec_module(mod)
    setattr(pkg, name.split('.')[-1], mod)
    return mod

got_mod = load('asi.graph_of_thought', 'src/graph_of_thought.py')
GraphOfThought = got_mod.GraphOfThought
AnalogicalReasoningDebugger = got_mod.AnalogicalReasoningDebugger
ReasoningHistoryLogger = load('asi.reasoning_history', 'src/reasoning_history.py').ReasoningHistoryLogger


class SimpleMemory:
    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = []
        self.meta = []
        self.compressor = types.SimpleNamespace(encoder=types.SimpleNamespace(in_features=dim))

    def add(self, vec: torch.Tensor, metadata=None):
        self.vectors.append(vec)
        self.meta.append(metadata[0] if metadata else None)

    def search(self, query: torch.Tensor, k: int = 5, *, mode: str = "standard", offset: torch.Tensor | None = None, **kwargs):
        if mode == "analogy" and offset is not None:
            query = query + offset
        sims = [torch.nn.functional.cosine_similarity(query, v, dim=0) for v in self.vectors]
        idx = sorted(range(len(sims)), key=lambda i: float(sims[i]), reverse=True)[:k]
        return torch.stack([self.vectors[i] for i in idx]), [self.meta[i] for i in idx]


class TestAnalogicalReasoningDebugger(unittest.TestCase):
    def test_detects_inconsistency(self):
        mem = SimpleMemory(3)

        vectors = {
            "man": torch.tensor([1.0, 0.0, 0.0]),
            "woman": torch.tensor([0.0, 1.0, 0.0]),
            "king": torch.tensor([1.0, 0.0, 1.0]),
            "queen": torch.tensor([0.0, 1.0, 1.0]),
            "princess": torch.tensor([0.0, 1.0, 2.0]),
        }
        for k, v in vectors.items():
            mem.add(v, metadata=[k])

        g = GraphOfThought()
        a = g.add_step("correct", metadata={"analogy": (vectors["king"], vectors["man"], vectors["woman"], "queen")})
        b = g.add_step("wrong", metadata={"analogy": (vectors["king"], vectors["man"], vectors["woman"], "princess")})

        logger = ReasoningHistoryLogger()
        dbg = AnalogicalReasoningDebugger(g, mem, logger)
        mism = dbg.check_steps()
        self.assertEqual(mism, [b])
        self.assertEqual(len(logger.get_history()), 1)
        self.assertIn("princess", logger.get_history()[0][1])


if __name__ == '__main__':  # pragma: no cover - test helper
    unittest.main()
