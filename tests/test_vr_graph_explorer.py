import importlib.machinery
import importlib.util
import sys
import types
import json
import http.client
from pathlib import Path
import unittest

pkg = types.ModuleType('asi')
sys.modules.setdefault('asi', pkg)
src_pkg = types.ModuleType('src')
src_pkg.__path__ = [str(Path('src'))]
sys.modules.setdefault('src', src_pkg)


def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod

GraphOfThought = _load('asi.graph_of_thought', 'src/graph_of_thought.py').GraphOfThought
ReasoningHistoryLogger = _load('asi.reasoning_history', 'src/reasoning_history.py').ReasoningHistoryLogger
VRGraphVisualizer = _load('asi.vr_graph_explorer', 'src/vr_graph_explorer.py').VRGraphVisualizer
VRGraphExplorer = _load('asi.vr_graph_explorer', 'src/vr_graph_explorer.py').VRGraphExplorer


class TestVRGraphExplorer(unittest.TestCase):
    def test_html(self):
        g = GraphOfThought()
        a = g.add_step('a')
        b = g.add_step('b')
        g.connect(a, b)
        vis = VRGraphVisualizer.from_graph(g)
        html = vis.to_html()
        self.assertIn('<html', html.lower())

    def test_server(self):
        g = GraphOfThought()
        a = g.add_step('a')
        b = g.add_step('b')
        g.connect(a, b)
        logger = ReasoningHistoryLogger()
        viewer = VRGraphExplorer(g, logger)
        viewer.start(port=0)
        port = viewer.port
        conn = http.client.HTTPConnection('localhost', port)
        conn.request('GET', '/graph/data')
        data = json.loads(conn.getresponse().read())
        self.assertEqual(len(data['nodes']), 2)
        viewer.stop()


if __name__ == '__main__':
    unittest.main()
