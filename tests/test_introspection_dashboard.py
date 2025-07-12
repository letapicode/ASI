import importlib.machinery
import importlib.util
import types
import sys
import json
import http.client
import unittest

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg


def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod

GraphOfThought = _load('asi.graph_of_thought', 'src/graph_of_thought.py').GraphOfThought
ReasoningHistoryLogger = _load('asi.reasoning_history', 'src/reasoning_history.py').ReasoningHistoryLogger
TelemetryLogger = _load('asi.telemetry', 'src/telemetry.py').TelemetryLogger
IntrospectionDashboard = _load('asi.dashboards', 'src/dashboards.py').IntrospectionDashboard


class TestIntrospectionDashboard(unittest.TestCase):
    def test_server(self):
        graph = GraphOfThought()
        a = graph.add_step('start', metadata={'timestamp': 0.0})
        b = graph.add_step('end', metadata={'timestamp': 1.0})
        graph.connect(a, b)

        history = ReasoningHistoryLogger()
        history.log('start -> end')

        tel = TelemetryLogger(interval=0.1)
        tel.start()
        dash = IntrospectionDashboard(graph, history, tel)
        dash.start(port=0)
        port = dash.port
        assert port is not None

        conn = http.client.HTTPConnection('localhost', port)
        conn.request('GET', '/json')
        data = json.loads(conn.getresponse().read())
        self.assertIn('graph', data)
        self.assertIn('history', data)
        self.assertIn('telemetry', data)

        conn.request('GET', '/history')
        hist = json.loads(conn.getresponse().read())
        self.assertEqual(hist[0][1], 'start -> end')

        dash.stop()
        tel.stop()


if __name__ == '__main__':
    unittest.main()

