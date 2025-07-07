import unittest
import importlib.machinery
import importlib.util
import sys
import types

pkg = types.ModuleType('asi')
pkg.__path__ = ['src']
pkg.__spec__ = importlib.machinery.ModuleSpec('asi', None, is_package=True)
sys.modules['asi'] = pkg


def _load(name: str, path: str):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    sys.modules[name] = mod
    return mod

class KnowledgeGraphMemory:
    def __init__(self) -> None:
        self.triples = []
        self.timestamps = []
    def add_triples(self, triples):
        for s, p, o, ts in triples:
            self.triples.append((s, p, o))
            self.timestamps.append(ts)
    def query_triples(self, **k):
        return []

KGVisualizer = _load('asi.kg_visualizer', 'src/kg_visualizer.py').KGVisualizer


class TestKGVisualizer(unittest.TestCase):
    def test_graph_json(self):
        kg = KnowledgeGraphMemory()
        kg.add_triples([('a', 'b', 'c', 1.0)])
        viz = KGVisualizer(kg)
        data = viz.graph_json()
        self.assertEqual(len(data['nodes']), 2)
        self.assertEqual(data['edges'][0]['predicate'], 'b')
        self.assertEqual(data['edges'][0]['timestamp'], 1.0)

    def test_server_lifecycle(self):
        kg = KnowledgeGraphMemory()
        viz = KGVisualizer(kg)
        viz.start(port=0)
        self.assertIsNotNone(viz.httpd)
        viz.stop()
        self.assertIsNone(viz.httpd)


if __name__ == '__main__':
    unittest.main()
