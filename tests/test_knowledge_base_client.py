import unittest
import types
import sys
import importlib.machinery
import importlib.util

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg
sys.modules['requests'] = types.ModuleType('requests')

loader = importlib.machinery.SourceFileLoader('asi.knowledge_base_client', 'src/knowledge_base_client.py')
spec = importlib.util.spec_from_loader('asi.knowledge_base_client', loader)
mod = importlib.util.module_from_spec(spec)
loader.exec_module(mod)
setattr(pkg, 'knowledge_base_client', mod)
KnowledgeBaseClient = mod.KnowledgeBaseClient


class DummyResp:
    def __init__(self, json_data):
        self._json = json_data
    def json(self):
        return self._json
    def raise_for_status(self):
        pass


class TestKnowledgeBaseClient(unittest.TestCase):
    def test_query(self):
        calls = {}
        def post(url, data=None, headers=None):
            calls['url'] = url
            return DummyResp({'results': {'bindings': [{'s': {'value': 'a'}, 'p': {'value': 'b'}, 'o': {'value': 'c'}}]}})
        sys.modules['requests'].post = post
        client = KnowledgeBaseClient('http://x')
        triples = client.query('select')
        self.assertEqual(triples[0], ('a', 'b', 'c'))
        self.assertEqual(calls['url'], 'http://x')


if __name__ == '__main__':
    unittest.main()
