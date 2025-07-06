import unittest
import http.client
import json
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

HierarchicalMemory = _load('asi.hierarchical_memory', 'src/hierarchical_memory.py').HierarchicalMemory
MemoryDashboard = _load('asi.memory_dashboard', 'src/memory_dashboard.py').MemoryDashboard
RetrievalVisualizer = _load('asi.retrieval_visualizer', 'src/retrieval_visualizer.py').RetrievalVisualizer


class TestRetrievalVisualizer(unittest.TestCase):
    def test_logging(self):
        mem = HierarchicalMemory(dim=2, compressed_dim=1, capacity=10)
        vis = RetrievalVisualizer(mem)
        vis.start()
        mem.search([0.0, 0.0], k=1)  # miss
        data = [[0.1, 0.2]]
        mem.add(data)
        mem.search(data[0], k=1)  # hit
        self.assertEqual(len(vis.log), 2)
        hits = [e["hit"] for e in vis.log]
        self.assertEqual(hits, [0.0, 1.0])
        vis.stop()

    def test_patterns_endpoint(self):
        mem = HierarchicalMemory(dim=2, compressed_dim=1, capacity=10)
        vis = RetrievalVisualizer(mem)
        vis.start()
        mem.add([[0.3, 0.4]])
        mem.search([0.3, 0.4], k=1)
        server = type("Stub", (), {"memory": mem, "telemetry": None})()
        dash = MemoryDashboard([server], visualizer=vis)
        dash.start(port=0)
        port = dash.port
        conn = http.client.HTTPConnection("localhost", port)
        conn.request("GET", "/patterns")
        resp = conn.getresponse()
        html = resp.read().decode()
        self.assertIn("<img", html)
        dash.stop()
        vis.stop()


if __name__ == '__main__':
    unittest.main()
