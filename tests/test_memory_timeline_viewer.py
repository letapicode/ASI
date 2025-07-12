import unittest
import importlib.machinery
import importlib.util
import types
import sys
import json
import http.client

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg


def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    sys.modules[name] = mod
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
