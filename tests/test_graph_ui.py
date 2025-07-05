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
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod

GraphOfThought = _load('asi.graph_of_thought', 'src/graph_of_thought.py').GraphOfThought
ReasoningHistoryLogger = _load('asi.reasoning_history', 'src/reasoning_history.py').ReasoningHistoryLogger
GraphUI = _load('asi.graph_ui', 'src/graph_ui.py').GraphUI


class TestGraphUI(unittest.TestCase):
    def test_endpoints(self):
        g = GraphOfThought()
        a = g.add_step('start')
        b = g.add_step('finish')
        g.connect(a, b)
        logger = ReasoningHistoryLogger()
        logger.log('summary')
        ui = GraphUI(g, logger)
        ui.start(port=0)
        port = ui.port
        conn = http.client.HTTPConnection('localhost', port)
        conn.request('GET', '/history')
        resp = conn.getresponse()
        hist = json.loads(resp.read())
        self.assertEqual(hist[0][1], 'summary')
        conn.request('GET', '/graph/data')
        resp = conn.getresponse()
        data = json.loads(resp.read())
        self.assertEqual(len(data['nodes']), 2)
        ui.stop()


if __name__ == '__main__':
    unittest.main()
