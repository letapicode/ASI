import unittest
import importlib.machinery
import importlib.util
import types
import sys
import json
import http.client
import time

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg


def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    sys.modules[name] = mod
    return mod

MultiAgentDashboard = _load('asi.multi_agent_dashboard', 'src/multi_agent_dashboard.py').MultiAgentDashboard
MultiAgentCoordinator = _load('asi.multi_agent_coordinator', 'src/multi_agent_coordinator.py').MultiAgentCoordinator
TelemetryLogger = _load('asi.telemetry', 'src/telemetry.py').TelemetryLogger
ReasoningHistoryLogger = _load('asi.reasoning_history', 'src/reasoning_history.py').ReasoningHistoryLogger


class StubAgent:
    def __init__(self, tel, hist):
        self.telemetry = tel
        self.history = hist

    def select_action(self, state):
        return 'act'

    def update(self, state, action, reward, next_state):
        pass

    def train(self, entries):
        pass


class TestMultiAgentDashboard(unittest.TestCase):
    def test_aggregate(self):
        tel = TelemetryLogger(interval=0.1, carbon_tracker=True)
        tel.start()
        hist = ReasoningHistoryLogger()
        hist.log('test')
        agent = StubAgent(tel, hist)
        coord = MultiAgentCoordinator({'a1': agent})
        coord.log.append(('a1', 'repo', 'act', 1.0))
        time.sleep(0.2)
        dash = MultiAgentDashboard(coord)
        data = dash.aggregate()
        tel.stop()
        self.assertIn('telemetry', data)
        self.assertIn('assignments', data)
        self.assertIn('reasoning', data)

    def test_http_server(self):
        tel = TelemetryLogger(interval=0.1)
        hist = ReasoningHistoryLogger()
        agent = StubAgent(tel, hist)
        coord = MultiAgentCoordinator({'a1': agent})
        dash = MultiAgentDashboard(coord)
        dash.start(port=0)
        port = dash.port
        conn = http.client.HTTPConnection('localhost', port)
        conn.request('GET', '/stats')
        resp = conn.getresponse()
        data = json.loads(resp.read())
        self.assertIn('assignments', data)
        dash.stop()


if __name__ == '__main__':
    unittest.main()
