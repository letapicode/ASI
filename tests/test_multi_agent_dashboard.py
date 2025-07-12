import unittest
import importlib.machinery
import importlib.util
import types
import sys
import json
import http.client
import time
np = types.SimpleNamespace(argmax=lambda x: 0)
sys.modules.setdefault('numpy', np)

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg
meta_stub = types.ModuleType('asi.meta_rl_refactor')
meta_stub.MetaRLRefactorAgent = type('A', (), {})
sys.modules['asi.meta_rl_refactor'] = meta_stub
rl_stub = types.ModuleType('asi.rl_decision_narrator')
rl_stub.RLDecisionNarrator = type('R', (), {})
sys.modules['asi.rl_decision_narrator'] = rl_stub
ct_stub = types.ModuleType('asi.carbon_tracker')
ct_stub.CarbonFootprintTracker = lambda *a, **k: None
sys.modules['asi.carbon_tracker'] = ct_stub


def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod

MultiAgentDashboard = _load('asi.dashboards', 'src/dashboards.py').MultiAgentDashboard
MultiAgentCoordinator = _load('asi.multi_agent_coordinator', 'src/multi_agent_coordinator.py').MultiAgentCoordinator
TelemetryLogger = _load('asi.telemetry', 'src/telemetry.py').TelemetryLogger
ReasoningHistoryLogger = _load('asi.reasoning_history', 'src/reasoning_history.py').ReasoningHistoryLogger
GraphOfThought = _load('asi.graph_of_thought', 'src/graph_of_thought.py').GraphOfThought


class StubAgent:
    def __init__(self, tel, hist, graph=None):
        self.telemetry = tel
        self.history = hist
        self.graph = graph

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
        g = GraphOfThought()
        g.add_step('start', metadata={'timestamp': 0.0})
        agent = StubAgent(tel, hist, g)
        coord = MultiAgentCoordinator({'a1': agent})
        coord.log.append(('a1', 'repo', 'act', 1.0))
        time.sleep(0.2)
        dash = MultiAgentDashboard(coord)
        data = dash.aggregate()
        tel.stop()
        self.assertIn('telemetry', data)
        self.assertIn('assignments', data)
        self.assertIn('reasoning', data)
        self.assertIn('merged_reasoning', data)

    def test_http_server(self):
        tel = TelemetryLogger(interval=0.1)
        hist = ReasoningHistoryLogger()
        g = GraphOfThought()
        g.add_step('s', metadata={'timestamp': 0.0})
        agent = StubAgent(tel, hist, g)
        coord = MultiAgentCoordinator({'a1': agent})
        dash = MultiAgentDashboard(coord)
        dash.start(port=0)
        port = dash.port
        conn = http.client.HTTPConnection('localhost', port)
        conn.request('GET', '/stats')
        resp = conn.getresponse()
        data = json.loads(resp.read())
        self.assertIn('assignments', data)
        self.assertIn('merged_reasoning', data)
        dash.stop()


if __name__ == '__main__':
    unittest.main()
