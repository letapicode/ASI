import tempfile
import types
import sys
import unittest
from unittest.mock import patch

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg
pkg.__path__ = ['src']


def _load(name, path):
    loader = __import__('importlib.machinery').machinery.SourceFileLoader(name, path)
    spec = __import__('importlib.util').util.spec_from_loader(name, loader)
    mod = __import__('importlib.util').util.module_from_spec(spec)
    mod.__package__ = name.rpartition('.')[0]
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod

kb_mod = _load('asi.knowledge_base_client', 'src/knowledge_base_client.py')
kg_mod = _load('asi.knowledge_graph_memory', 'src/knowledge_graph_memory.py')
KnowledgeBaseClient = kb_mod.KnowledgeBaseClient
load_kb_config = kb_mod.load_kb_config
KnowledgeGraphMemory = kg_mod.KnowledgeGraphMemory
TimedTriple = kg_mod.TimedTriple


class TestKnowledgeBaseClient(unittest.TestCase):
    def test_query_and_cache(self):
        client = KnowledgeBaseClient('http://kb', cache_size=1)
        reply = {
            'results': {
                'bindings': [
                    {'s': {'value': 'a'}, 'p': {'value': 'b'}, 'o': {'value': 'c'}}
                ]
            }
        }
        resp = types.SimpleNamespace(json=lambda: reply, raise_for_status=lambda: None)
        with patch('requests.post', return_value=resp) as post:
            res1 = client.query('Q')
            res2 = client.query('Q')
        self.assertEqual(res1, [('a', 'b', 'c')])
        self.assertEqual(res2, [('a', 'b', 'c')])
        post.assert_called_once()

    def test_load_config(self):
        with tempfile.NamedTemporaryFile('w+', delete=False) as f:
            f.write('endpoint: http://ex\ncache_size: 2\n')
            name = f.name
        cfg = load_kb_config(name)
        self.assertEqual(cfg['endpoint'], 'http://ex')
        self.assertEqual(cfg['cache_size'], 2)

    def test_kg_remote_fallback(self):
        kg = KnowledgeGraphMemory()
        kg.add_triples([('x', 'y', 'z')])
        client = KnowledgeBaseClient('http://kb')
        resp = types.SimpleNamespace(json=lambda: {
            'results': {
                'bindings': [
                    {'s': {'value': 'a'}, 'p': {'value': 'b'}, 'o': {'value': 'c'}}
                ]
            }
        }, raise_for_status=lambda: None)
        # local hit should not call requests
        with patch('requests.post') as post:
            out = kg.query_triples(subject='x', kb_client=client)
            post.assert_not_called()
            self.assertEqual(len(out), 1)
        # miss triggers remote call
        with patch('requests.post', return_value=resp) as post:
            out = kg.query_triples(subject='a', kb_client=client)
            post.assert_called_once()
            self.assertEqual(len(out), 1)
            t = out[0]
            self.assertIsInstance(t, TimedTriple)
            self.assertEqual((t.subject, t.predicate, t.object), ('a', 'b', 'c'))


if __name__ == '__main__':
    unittest.main()
