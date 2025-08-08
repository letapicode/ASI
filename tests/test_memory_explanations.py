import unittest
import json
import http.client
import importlib.machinery
import importlib.util
import types
import sys
try:
    import torch
except Exception:  # pragma: no cover - torch may be unavailable
    class _Tensor(list):
        @property
        def ndim(self):
            return 2 if self and isinstance(self[0], list) else 1

        def unsqueeze(self, dim):
            return _Tensor([self])

        def tolist(self):
            return list(self)

    def _randn(*shape):
        if not shape:
            return _Tensor([])
        if len(shape) == 1:
            return _Tensor([[0.0 for _ in range(shape[0])]])
        rows = shape[0]
        cols = shape[1]
        return _Tensor([[0.0 for _ in range(cols)] for _ in range(rows)])

    torch = types.SimpleNamespace(
        randn=_randn,
        zeros=lambda *a, **k: _Tensor([[0.0 for _ in range(a[1])] for _ in range(a[0])]) if a else _Tensor([]),
        ones=lambda *a, **k: _Tensor([[1.0 for _ in range(a[1])] for _ in range(a[0])]) if a else _Tensor([]),
        tensor=lambda v, dtype=None: _Tensor(v),
        nn=types.SimpleNamespace(Module=object),
        Tensor=_Tensor,
        float32=object,
    )
    sys.modules['torch'] = torch

    np = types.SimpleNamespace(
        mean=lambda x: sum(x)/len(x) if x else 0.0,
        corrcoef=lambda a, b: [[0, 0], [0, 0]],
        array=lambda x: x,
        ndarray=list,
    )
    sys.modules['numpy'] = np
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
    psutil_stub = types.SimpleNamespace(
        cpu_percent=lambda interval=None: 0.0,
        virtual_memory=lambda: types.SimpleNamespace(percent=0.0),
        net_io_counters=lambda: types.SimpleNamespace(bytes_sent=0, bytes_recv=0),
    )
    sys.modules['psutil'] = psutil_stub
    sys.modules['requests'] = types.ModuleType('requests')
    sys.modules['PIL'] = types.ModuleType('PIL')
    sys.modules['PIL.Image'] = types.ModuleType('PIL.Image')
    plt = types.SimpleNamespace(
        subplots=lambda *a, **k: (types.SimpleNamespace(savefig=lambda *a, **k: None), [types.SimpleNamespace(plot=lambda *a, **k: None, set_ylabel=lambda *a, **k: None, set_xlabel=lambda *a, **k: None, imshow=lambda *a, **k: None) for _ in range((a[0] if a else 1))]),
        close=lambda *a, **k: None,
        tight_layout=lambda *a, **k: None,
    )
    sys.modules['matplotlib'] = types.ModuleType('matplotlib')
    sys.modules['matplotlib.pyplot'] = plt
    retrieval_saliency_stub = types.ModuleType('asi.retrieval_saliency')
    retrieval_saliency_stub.token_saliency = lambda q, r: []
    retrieval_saliency_stub.image_saliency = lambda q, r: []
    sys.modules['asi.retrieval_saliency'] = retrieval_saliency_stub
    tc_stub = types.ModuleType('asi.transformer_circuits')
    tc_stub.AttentionVisualizer = type('AV', (), {})
    sys.modules['asi.transformer_circuits'] = tc_stub

pkg = types.ModuleType('asi')
pkg.__path__ = ['src']
pkg.__spec__ = importlib.machinery.ModuleSpec('asi', None, is_package=True)
sys.modules['asi'] = pkg


def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    sys.modules[name] = mod
    return mod

re_mod = _load('asi.retrieval_analysis', 'src/retrieval_analysis.py')
RetrievalExplainer = re_mod.RetrievalExplainer


class HierarchicalMemory:
    def __init__(self, *a, **k):
        self.data = []
        self.meta = []
        self.last_trace = None

    def add(self, v, metadata=None):
        if isinstance(v, list) and not isinstance(v, torch.Tensor):
            self.data.extend(v)
        else:
            self.data.append(v)
        if metadata:
            self.meta.extend(metadata)

    def search(
        self,
        q,
        k=1,
        return_scores=False,
        return_provenance=False,
        return_summary=False,
    ):
        vecs = self.data[:k]
        metas = self.meta[:k]
        scores = [1.0 for _ in vecs]
        q_t = torch.tensor(q)
        vecs_t = torch.tensor(vecs)
        summary = ""
        if return_summary:
            summary = RetrievalExplainer.summarize(q_t, vecs_t, scores, metas)
        self.last_trace = {
            "query": q_t,
            "results": vecs_t,
            "scores": scores,
            "provenance": metas,
        }
        if summary:
            self.last_trace["summary"] = summary
        extras = []
        if return_scores:
            extras.append(scores)
        if return_provenance:
            extras.append(metas)
        if return_summary:
            extras.append(summary)
        return (vecs, metas, *extras) if extras else (vecs, metas)

stub_hm = types.ModuleType('asi.hierarchical_memory')
stub_hm.HierarchicalMemory = HierarchicalMemory
class MemoryServer:
    def __init__(self, memory, address=None, max_workers=4, telemetry=None):
        self.memory = memory
        self.telemetry = telemetry
    def start(self):
        pass
    def stop(self, grace=0):
        pass
stub_hm.MemoryServer = MemoryServer
sys.modules['asi.hierarchical_memory'] = stub_hm

MemoryDashboard = _load('asi.dashboards', 'src/dashboards.py').MemoryDashboard

class TestMemoryExplanations(unittest.TestCase):
    def test_summary_return_and_dashboard(self):
        mem = HierarchicalMemory(dim=2, compressed_dim=1, capacity=10)
        data = torch.randn(1, 2)
        mem.add(data, metadata=["item"])
        vec, meta, scores, prov, summary = mem.search(
            data[0], k=1,
            return_scores=True,
            return_provenance=True,
            return_summary=True,
        )
        self.assertIsInstance(summary, str)
        self.assertIn("item", summary)
        self.assertEqual(mem.last_trace["summary"], summary)

        server = type("Stub", (), {"memory": mem, "telemetry": None})()
        dash = MemoryDashboard([server])
        dash.start(port=0)
        port = dash.port
        conn = http.client.HTTPConnection("localhost", port)
        conn.request("GET", "/trace")
        resp = conn.getresponse()
        data = json.loads(resp.read())
        dash.stop()
        self.assertEqual(data["summary"], summary)

    def test_no_summary_when_disabled(self):
        mem = HierarchicalMemory(dim=2, compressed_dim=1, capacity=10)
        data = torch.randn(1, 2)
        mem.add(data, metadata=["foo"])
        vec, meta = mem.search(data[0], k=1)
        self.assertIsInstance(vec, list)
        self.assertNotIn("summary", mem.last_trace)

        server = type("Stub", (), {"memory": mem, "telemetry": None})()
        dash = MemoryDashboard([server])
        dash.start(port=0)
        port = dash.port
        conn = http.client.HTTPConnection("localhost", port)
        conn.request("GET", "/trace")
        resp = conn.getresponse()
        data = json.loads(resp.read())
        dash.stop()
        self.assertIsInstance(data.get("summary"), str)

if __name__ == "__main__":
    unittest.main()
