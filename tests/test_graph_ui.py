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
        ui = GraphUI(g, logger)
        ui.start(port=0)
        port = ui.port
        conn = http.client.HTTPConnection('localhost', port)
        # add node
        body = json.dumps({'text': 'mid'})
        conn.request('POST', '/graph/node', body, {'Content-Type': 'application/json'})
        resp = conn.getresponse()
        node_id = json.loads(resp.read())['id']
        conn.request('POST', '/graph/edge', json.dumps({'src': a, 'dst': node_id}),
                     {'Content-Type': 'application/json'})
        conn.getresponse().read()
        conn.request('POST', '/graph/recompute')
        summary = json.loads(conn.getresponse().read())['summary']
        self.assertIn('start', summary)
        conn.request('GET', '/graph/data')
        resp = conn.getresponse()
        data = json.loads(resp.read())
        self.assertEqual(len(data['nodes']), 3)
        conn.request('GET', '/history')
        resp = conn.getresponse()
        hist = json.loads(resp.read())
        self.assertGreaterEqual(len(hist), 2)
        ui.stop()


if __name__ == '__main__':
    unittest.main()
