import unittest
import importlib.machinery
import importlib.util
import types
import sys

src_pkg = types.ModuleType('src')
sys.modules['src'] = src_pkg

loader = importlib.machinery.SourceFileLoader('src.research_ingest', 'src/research_ingest.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
ri = importlib.util.module_from_spec(spec)
ri.__package__ = 'src'
sys.modules['src.research_ingest'] = ri
loader.exec_module(ri)


class TestResearchIngest(unittest.TestCase):
    def test_suggest_modules(self):
        papers = [{'title': 'Reinforcement Learning for Memory Models'}]
        mods = ri.suggest_modules(papers)
        self.assertIn('world_model_rl', mods)
        self.assertIn('hierarchical_memory', mods)


if __name__ == '__main__':
    unittest.main()
