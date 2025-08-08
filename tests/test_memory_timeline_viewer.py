import unittest
import importlib.machinery
import importlib.util
import types
import sys
import json
import http.client

pkg = types.ModuleType('asi')
pkg.__path__ = ['src']
pkg.__spec__ = importlib.machinery.ModuleSpec('asi', None, is_package=True)
sys.modules['asi'] = pkg

ra_stub = types.ModuleType('asi.retrieval_analysis')
ra_stub.RetrievalExplainer = type('RE', (), {})
ra_stub.RetrievalVisualizer = type('RV', (), {})
sys.modules['asi.retrieval_analysis'] = ra_stub
tc_stub = types.ModuleType('asi.transformer_circuits')
tc_stub.AttentionVisualizer = type('AV', (), {})
sys.modules['asi.transformer_circuits'] = tc_stub
plt = types.SimpleNamespace(
    subplots=lambda *a, **k: (types.SimpleNamespace(savefig=lambda *a, **k: None), [types.SimpleNamespace(plot=lambda *a, **k: None, set_ylabel=lambda *a, **k: None, set_xlabel=lambda *a, **k: None, imshow=lambda *a, **k: None) for _ in range((a[0] if a else 1))]),
    close=lambda *a, **k: None,
    tight_layout=lambda *a, **k: None,
)
sys.modules['matplotlib'] = types.ModuleType('matplotlib')
sys.modules['matplotlib.pyplot'] = plt


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

MemoryTimelineViewer = _load('asi.memory_timeline_viewer', 'src/memory_timeline_viewer.py').MemoryTimelineViewer
TelemetryLogger = _load('asi.telemetry', 'src/telemetry.py').TelemetryLogger
MemoryEventDetector = _load('asi.memory_event_detector', 'src/memory_event_detector.py').MemoryEventDetector
HierarchicalMemory = _load('asi.hierarchical_memory', 'src/hierarchical_memory.py').HierarchicalMemory
MemoryDashboard = _load('asi.dashboards', 'src/dashboards.py').MemoryDashboard
serve = _load('asi.memory_service', 'src/memory_service.py').serve


class TestMemoryTimelineViewer(unittest.TestCase):
    def test_json_output(self):
        tel = TelemetryLogger(interval=0.1)
        entry = {"hits": 1, "misses": 1, "latency": 0.2, "carbon_intensity": 0.5}
        tel.history.append(entry)
        tel.event_detector.update(entry)
        viewer = MemoryTimelineViewer(tel)
        data = json.loads(viewer.to_json())
        self.assertEqual(len(data["timeline"]), 1)
        self.assertEqual(data["timeline"][0]["hit_rate"], 0.5)
        self.assertTrue(data["events"])  # event recorded

    def test_dashboard_integration(self):
        mem = HierarchicalMemory(dim=2, compressed_dim=1, capacity=10)
        tel = TelemetryLogger(interval=0.1)
        entry = {"hits": 1, "misses": 0, "latency": 0.1, "carbon_intensity": 0.4}
        tel.history.append(entry)
        tel.event_detector.update(entry)
        server = serve(mem, 'localhost:50920', telemetry=tel)
        dash = MemoryDashboard([server])
        dash.start(port=0)
        port = dash.port
        conn = http.client.HTTPConnection('localhost', port)
        conn.request('GET', '/timeline')
        resp = conn.getresponse()
        data = json.loads(resp.read())
        dash.stop()
        server.stop(0)
        self.assertEqual(len(data['timeline']), 1)
        self.assertEqual(data['timeline'][0]['carbon_intensity'], 0.4)


if __name__ == '__main__':
    unittest.main()
