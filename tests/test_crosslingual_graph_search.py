import importlib.machinery
import importlib.util
import types
import sys
import unittest
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch optional
    raise unittest.SkipTest("torch not available")

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg


def load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'asi'
    sys.modules[name] = mod
    loader.exec_module(mod)
    setattr(pkg, name.split('.')[-1], mod)
    return mod

di = load('asi.data_ingest', 'src/data_ingest.py')
goth = load('asi.graph_of_thought', 'src/graph_of_thought.py')
cg = load('asi.cross_lingual_graph', 'src/cross_lingual_graph.py')

CrossLingualReasoningGraph = cg.CrossLingualReasoningGraph
CrossLingualTranslator = di.CrossLingualTranslator


class TestCrossLingualGraphSearch(unittest.TestCase):
    def test_search_ranking(self):
        tr = CrossLingualTranslator(['es'])
        g = CrossLingualReasoningGraph(translator=tr)
        n0 = g.add_step('hello')
        n1 = g.add_step('goodbye')
        result = g.search('hola', 'es')

        def embed(text):
            seed = abs(hash(text)) % (2 ** 32)
            rng = np.random.default_rng(seed)
            return rng.standard_normal(8)

        def cos(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

        q_vec = embed('hola')
        sims = {
            n0: cos(embed(tr.translate('hello', 'es')), q_vec),
            n1: cos(embed(tr.translate('goodbye', 'es')), q_vec),
        }
        expected = [nid for nid, _ in sorted(sims.items(), key=lambda kv: kv[1], reverse=True)]
        self.assertEqual(result, expected)
        self.assertIn('es', g.nodes[n0].metadata['embeddings'])
        self.assertIn('es', g.nodes[n1].metadata['embeddings'])


if __name__ == '__main__':  # pragma: no cover - test helper
    unittest.main()
