import unittest
import tempfile
from pathlib import Path
import types
import sys
import importlib.machinery
import importlib.util

# minimal stubs
np = types.SimpleNamespace(
    mean=lambda x: sum(x) / len(x) if x else 0.0,
    corrcoef=lambda a, b: [[0, 0], [0, 0]],
    array=lambda x: x,
    ndarray=list,
)
sys.modules['numpy'] = np
sys.modules["PIL.PngImagePlugin"] = types.ModuleType("PIL.PngImagePlugin")
sys.modules["matplotlib"] = types.ModuleType("matplotlib")
plt = types.SimpleNamespace(subplots=lambda *a, **k: (types.SimpleNamespace(savefig=lambda *a, **k: None), [types.SimpleNamespace(plot=lambda *a, **k: None, set_ylabel=lambda *a, **k: None, set_xlabel=lambda *a, **k: None, imshow=lambda *a, **k: None) for _ in range(1)]), close=lambda *a, **k: None, tight_layout=lambda *a, **k: None)
sys.modules["matplotlib.pyplot"] = plt
sys.modules["PIL.UnidentifiedImageError"] = type("E", (), {})
sys.modules["PIL"] = types.ModuleType("PIL")
stub_hm = types.ModuleType("asi.hierarchical_memory")
class MemoryServer:
    def __init__(self, *a, **k):
        self.memory = a[0]
        self.telemetry = k.get("telemetry")
    def start(self):
        pass
    def stop(self, grace=0):
        pass
stub_hm.MemoryServer = MemoryServer
sys.modules["asi.hierarchical_memory"] = stub_hm
telemetry_stub = types.ModuleType("asi.telemetry")
class TL:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.metrics = {}
    def get_stats(self):
        return {}
    def get_events(self):
        return []
    def record_trust(self, score: float):
        self.metrics["trust_score"] = score
telemetry_stub.TelemetryLogger = TL
sys.modules["asi.telemetry"] = telemetry_stub
sys.modules["PIL.Image"] = types.ModuleType("PIL.Image")
# make src importable as 'asi'
pkg = types.ModuleType('asi')
pkg.__path__ = ['src']
pkg.__spec__ = importlib.machinery.ModuleSpec('asi', None, is_package=True)
sys.modules['asi'] = pkg
class _Tensor(list):
    def unsqueeze(self, dim):
        return _Tensor([self])
    @property
    def ndim(self):
        return 1

torch = types.SimpleNamespace(tensor=lambda v, dtype=None: _Tensor(v), Tensor=_Tensor, float32=object)
sys.modules['torch'] = torch

from asi.dataset_lineage import DatasetLineageManager
from asi.blockchain_provenance_ledger import BlockchainProvenanceLedger
from asi.retrieval_trust_scorer import RetrievalTrustScorer
from asi.memory_dashboard import MemoryDashboard


class DummyMemory:
    def __init__(self) -> None:
        self.data = []
        self.meta = []
        self.hit_count = 0
        self.miss_count = 0
        self.last_trace = None

    def add(self, vec, metadata=None):
        self.data.append(vec)
        if metadata:
            self.meta.extend(metadata)

    def search(self, q, k=1):
        if self.data:
            self.hit_count += 1
            res = self.data[:k]
            meta = self.meta[:k]
            self.last_trace = {
                "query": q,
                "results": res,
                "scores": [1.0] * len(res),
                "provenance": meta,
            }
            return res, meta
        self.miss_count += 1
        self.last_trace = None
        return [], []

    def get_stats(self):
        return {"hits": self.hit_count, "misses": self.miss_count}

    def __len__(self):
        return len(self.data)


class StubTelemetry:
    def __init__(self) -> None:
        self.metrics = {}

    def get_stats(self):
        return {}

    def get_events(self):
        return []

    def record_trust(self, score: float) -> None:
        self.metrics['trust_score'] = score


class TestRetrievalTrustScorer(unittest.TestCase):
    def test_score_and_logging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lineage = DatasetLineageManager(root)
            ledger = BlockchainProvenanceLedger(root)
            lineage.ledger = ledger
            f = root / 'a.txt'
            f.write_text('x')
            lineage.record([], [f], note='add')

            mem = DummyMemory()
            mem.add([0, 1], metadata=[str(f)])
            telemetry = StubTelemetry()
            server = type('S', (), {'memory': mem, 'telemetry': telemetry})()

            scorer = RetrievalTrustScorer(lineage, ledger)
            dash = MemoryDashboard([server], trust_scorer=scorer)
            mem.search([0, 1], k=1)
            stats = dash.aggregate()
            dash.to_html()
            self.assertIn('trust_score', stats)
            self.assertGreater(stats['trust_score'], 0.0)
            self.assertIn('trust_score', telemetry.metrics)


if __name__ == '__main__':
    unittest.main()
