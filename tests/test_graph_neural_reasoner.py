import unittest
import importlib.machinery
import importlib.util
import types
import sys

src_pkg = types.ModuleType('src')
src_pkg.__path__ = ['src']
src_pkg.__spec__ = importlib.machinery.ModuleSpec('src', None, is_package=True)
sys.modules['src'] = src_pkg
for mod_name in ['graph_neural_reasoner', 'knowledge_graph_memory']:
    loader = importlib.machinery.SourceFileLoader(f'src.{mod_name}', f'src/{mod_name}.py')
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'src'
    loader.exec_module(mod)
    sys.modules[f'src.{mod_name}'] = mod

GraphNeuralReasoner = sys.modules['src.graph_neural_reasoner'].GraphNeuralReasoner
KnowledgeGraphMemory = sys.modules['src.knowledge_graph_memory'].KnowledgeGraphMemory

class TestGraphNeuralReasoner(unittest.TestCase):
    def test_predict(self):
        kg = KnowledgeGraphMemory()
        kg.add_triples([('a','r','b'), ('b','r','c')])
        reasoner = GraphNeuralReasoner(kg)
        p = reasoner.predict_link('a','b')
        q = reasoner.predict_link('a','c')
        self.assertGreaterEqual(p, q)

if __name__ == '__main__':
    unittest.main()
