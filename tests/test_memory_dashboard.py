import unittest
import torch
import time
import json
import http.client
import importlib.machinery
import importlib.util
import types
import sys

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


if __name__ == "__main__":
    unittest.main()
