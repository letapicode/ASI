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
sys.modules['asi'] = pkg

def _load(name, path):
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
        mem = HierarchicalMemory(dim=2, compressed_dim=1, capacity=10)
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
        mem = HierarchicalMemory(dim=2, compressed_dim=1, capacity=10)
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
