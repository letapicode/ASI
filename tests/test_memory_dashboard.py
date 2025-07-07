import unittest
import time
import json
import http.client
import importlib.machinery
import importlib.util
import types
import sys

# provide minimal torch stub
class _Tensor(list):
    @property
    def ndim(self):
        return 2 if self and isinstance(self[0], list) else 1
    def unsqueeze(self, dim):
        return _Tensor([self])

torch = types.SimpleNamespace(
    randn=lambda *a, **k: _Tensor([[0.0] * (a[-1] if len(a) > 1 else 1)]),
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
retrieval_explainer_stub = types.ModuleType('asi.retrieval_explainer')
class _RE:
    @staticmethod
    def format(*a, **k):
        return []
    @staticmethod
    def summarize(*a, **k):
        return ''
    @staticmethod
    def summarize_multimodal(q, r, scores, provenance):
        parts = []
        for i, (score, prov) in enumerate(zip(scores, provenance), start=1):
            if isinstance(prov, dict):
                fields = []
                if 'text' in prov:
                    fields.append(str(prov['text']))
                if 'image' in prov:
                    fields.append(prov['image'])
                if 'audio' in prov:
                    fields.append(prov['audio'])
                src = ', '.join(fields)
            else:
                src = str(prov)
            parts.append(f"{i}. {src} (score={score:.3f})")
        return ' | '.join(parts)
retrieval_explainer_stub.RetrievalExplainer = _RE
sys.modules['asi.retrieval_explainer'] = retrieval_explainer_stub
retrieval_visualizer_stub = types.ModuleType('asi.retrieval_visualizer')
class RV:
    def __init__(self, *a, **k):
        pass
    def pattern_image(self):
        return ''
retrieval_visualizer_stub.RetrievalVisualizer = RV
sys.modules['asi.retrieval_visualizer'] = retrieval_visualizer_stub
memory_timeline_stub = types.ModuleType('asi.memory_timeline_viewer')
class MTV:
    def __init__(self, *a, **k):
        pass
    def to_json(self):
        return '{}'
memory_timeline_stub.MemoryTimelineViewer = MTV
sys.modules['asi.memory_timeline_viewer'] = memory_timeline_stub
stub_hm = types.ModuleType('asi.hierarchical_memory')
class _HM:
    def __init__(self, *a, **k):
        self.data = []
        self.meta = []
        self.kg = None
        self.hit_count = 0
        self.miss_count = 0
        self.store = self
        self._meta = self.meta
    def add(self, v, metadata=None):
        self.data.append(v)
        if metadata:
            self.meta.extend(metadata)
    def search(self, q, k=1):
        if self.data:
            self.hit_count += 1
            return self.data[:k], self.meta[:k]
        self.miss_count += 1
        return [], []
    def delete(self, index=None, tag=None):
        self.data.clear(); self.meta.clear()
    def __len__(self):
        return len(self.data)
    def get_stats(self):
        return {'hits': self.hit_count, 'misses': self.miss_count}
class _MS:
    def __init__(self, memory, address=None, max_workers=4, telemetry=None):
        self.memory = memory
        self.telemetry = telemetry
    def start(self):
        pass
    def stop(self, grace=0):
        pass
stub_hm.HierarchicalMemory = _HM
stub_hm.MemoryServer = _MS
sys.modules['asi.hierarchical_memory'] = stub_hm
telemetry_stub = types.ModuleType('asi.telemetry')
class TL:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.history = []
        self.event_detector = types.SimpleNamespace()
    def get_stats(self):
        return {}
    def get_events(self):
        return []
telemetry_stub.TelemetryLogger = TL
sys.modules['asi.telemetry'] = telemetry_stub

pkg = types.ModuleType('asi')
pkg.__path__ = ['src']
pkg.__spec__ = importlib.machinery.ModuleSpec('asi', None, is_package=True)
sys.modules['asi'] = pkg

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
sys.modules['requests'] = types.ModuleType('requests')
psutil_stub = types.SimpleNamespace(
    cpu_percent=lambda interval=None: 0.0,
    virtual_memory=lambda: types.SimpleNamespace(percent=0.0),
    net_io_counters=lambda: types.SimpleNamespace(bytes_sent=0, bytes_recv=0),
)
sys.modules['psutil'] = psutil_stub

def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    if name == 'asi.hierarchical_memory':
        mod = types.ModuleType(name)
        class HierarchicalMemory:
            def __init__(self, *a, **k):
                self.data = []
                self.meta = []
                self.kg = None
            def add(self, v, metadata=None):
                if isinstance(v, list):
                    self.data.extend(v)
                else:
                    self.data.append(v)
                if metadata:
                    self.meta.extend(metadata)
            def search(self, q, k=1):
                return self.data[:k], self.meta[:k]
            def delete(self, index=None, tag=None):
                self.data.clear()
                self.meta.clear()
            def __len__(self):
                return len(self.data)
            def get_stats(self):
                return {'hits': 0, 'misses': 0}
        class MemoryServer:
            def __init__(self, memory, address=None, max_workers=4, telemetry=None):
                self.memory = memory
                self.telemetry = telemetry
            def start(self):
                pass
            def stop(self, grace=0):
                pass
        mod.HierarchicalMemory = HierarchicalMemory
        mod.MemoryServer = MemoryServer
        sys.modules[name] = mod
        return mod
    if name == 'asi.streaming_compression':
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        return mod
    if name == 'asi.memory_service':
        hm = _load('asi.hierarchical_memory', 'src/hierarchical_memory.py')
        mod = types.ModuleType(name)
        def serve(memory, address, max_workers=4, telemetry=None):
            server = hm.MemoryServer(memory, address=address, max_workers=max_workers, telemetry=telemetry)
            server.start()
            return server
        mod.serve = serve
        mod.MemoryServer = hm.MemoryServer
        sys.modules[name] = mod
        return mod
    if name == 'asi.retrieval_saliency':
        mod = types.ModuleType(name)
        mod.token_saliency = lambda q, r: []
        mod.image_saliency = lambda q, r: []
        sys.modules[name] = mod
        return mod
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    sys.modules[name] = mod
    return mod

MemoryDashboard = _load('asi.memory_dashboard', 'src/memory_dashboard.py').MemoryDashboard
_load('asi.streaming_compression', 'src/streaming_compression.py')
HierarchicalMemory = _load('asi.hierarchical_memory', 'src/hierarchical_memory.py').HierarchicalMemory
serve = _load('asi.memory_service', 'src/memory_service.py').serve
TelemetryLogger = _load('asi.telemetry', 'src/telemetry.py').TelemetryLogger


class TestMemoryDashboard(unittest.TestCase):
    def test_aggregate(self):
        mem = HierarchicalMemory(dim=2, compressed_dim=1, capacity=10, encryption_key=b'0'*16)
        logger = TelemetryLogger(interval=0.1)
        server = serve(mem, "localhost:50910", telemetry=logger)
        mem.add(torch.randn(1, 2))
        mem.search(torch.randn(2), k=1)
        dashboard = MemoryDashboard([server])
        time.sleep(0.2)
        stats = dashboard.aggregate()
        server.stop(0)
        self.assertIn("hit_rate", stats)

    def test_http_endpoints(self):
        mem = HierarchicalMemory(dim=2, compressed_dim=1, capacity=10, encryption_key=b'0'*16)
        server = type("Stub", (), {"memory": mem, "telemetry": None})()
        dash = MemoryDashboard([server])
        dash.start(port=0)
        port = dash.port
        conn = http.client.HTTPConnection("localhost", port)
        vec = [0.1, 0.2]
        conn.request("POST", "/add", body=json.dumps({"vector": vec, "metadata": "m1"}))
        conn.getresponse().read()
        conn.request("GET", "/entries")
        resp = conn.getresponse()
        entries = json.loads(resp.read())
        self.assertEqual(entries[0]["meta"], "m1")
        conn.request("DELETE", "/delete", body=json.dumps({"index": 0}))
        conn.getresponse().read()
        self.assertEqual(len(mem), 0)
        dash.stop()

    def test_trace_summary_multimodal(self):
        mem = HierarchicalMemory(dim=2, compressed_dim=1, capacity=10, encryption_key=b'0'*16)
        mem.last_trace = {
            "query": [0, 0],
            "results": [[1, 1]],
            "scores": [0.5],
            "provenance": [{"text": "hello", "image": "img.png", "audio": "a.wav"}],
        }
        server = type("Stub", (), {"memory": mem, "telemetry": None})()
        dash = MemoryDashboard([server])
        dash.start(port=0)
        port = dash.port
        conn = http.client.HTTPConnection("localhost", port)
        conn.request("GET", "/trace")
        resp = conn.getresponse()
        data = json.loads(resp.read())
        dash.stop()
        self.assertIn("img.png", data["summary"])


if __name__ == "__main__":
    unittest.main()
