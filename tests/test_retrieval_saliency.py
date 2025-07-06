import unittest
import importlib.machinery
import importlib.util
import types
import sys
import http.client
import torch

pkg = types.ModuleType('asi')
pkg.__path__ = ['src']
pkg.__spec__ = importlib.machinery.ModuleSpec('asi', None, is_package=True)
sys.modules['asi'] = pkg

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
sys.modules['requests'] = types.ModuleType('requests')
psutil_stub = types.SimpleNamespace(
    cpu_percent=lambda interval=None: 0.0,
    virtual_memory=lambda: types.SimpleNamespace(percent=0.0),
    net_io_counters=lambda: types.SimpleNamespace(bytes_sent=0, bytes_recv=0),
)
sys.modules['psutil'] = psutil_stub


def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    sys.modules[name] = mod
    if name == 'asi.hierarchical_memory' and not hasattr(mod, 'MemoryServer'):
        class MemoryServer:
            def __init__(self, memory, address=None, max_workers=4, telemetry=None):
                self.memory = memory
                self.telemetry = telemetry
            def start(self):
                pass
            def stop(self, grace=0):
                pass
        mod.MemoryServer = MemoryServer
    return mod

RetrievalSaliency = _load('asi.retrieval_saliency', 'src/retrieval_saliency.py')
_load('asi.streaming_compression', 'src/streaming_compression.py')
_load('asi.retrieval_explainer', 'src/retrieval_explainer.py')
HierarchicalMemory = _load('asi.hierarchical_memory', 'src/hierarchical_memory.py').HierarchicalMemory
MemoryDashboard = _load('asi.memory_dashboard', 'src/memory_dashboard.py').MemoryDashboard
RetrievalVisualizer = _load('asi.retrieval_visualizer', 'src/retrieval_visualizer.py').RetrievalVisualizer


class TestRetrievalSaliency(unittest.TestCase):
    def test_shapes(self):
        tokens = torch.randn(3, 4)
        res = torch.randn(2, 4)
        sal = RetrievalSaliency.token_saliency(tokens, res)
        self.assertEqual(sal.shape, (2, 3))

        img = torch.randn(3, 2, 2)
        res2 = torch.randn(2, 12)
        sal2 = RetrievalSaliency.image_saliency(img, res2)
        self.assertEqual(sal2.shape, (2, 2, 2))

    def test_dashboard_integration(self):
        mem = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10, encryption_key=b'0'*16)
        vis = RetrievalVisualizer(mem)
        vis.start()
        vec = torch.randn(1, 4)
        mem.add(vec)
        mem.search(vec[0], k=1)
        self.assertTrue(len(vis.saliencies) > 0)
        server = type('Stub', (), {'memory': mem, 'telemetry': None})()
        dash = MemoryDashboard([server], visualizer=vis)
        dash.start(port=0)
        port = dash.port
        conn = http.client.HTTPConnection('localhost', port)
        conn.request('GET', '/patterns')
        resp = conn.getresponse()
        html = resp.read().decode()
        self.assertIn('<img', html)
        dash.stop()
        vis.stop()


if __name__ == '__main__':
    unittest.main()
