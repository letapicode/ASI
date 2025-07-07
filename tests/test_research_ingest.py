import unittest
import importlib.machinery
import importlib.util
import types
import sys


src_pkg = types.ModuleType('src')
sys.modules['src'] = src_pkg
sys.modules['numpy'] = types.ModuleType('numpy')
sys.modules['torch'] = types.ModuleType('torch')
sys.modules['requests'] = types.ModuleType('requests')
sys.modules['aiohttp'] = types.ModuleType('aiohttp')
sys.modules['PIL'] = types.ModuleType('PIL')
sys.modules['PIL.Image'] = types.ModuleType('PIL.Image')
sys.modules['src.carbon_tracker'] = types.SimpleNamespace(
    CarbonFootprintTracker=type('CFT', (), {'__init__': lambda self, **kw: None})
)

di_loader = importlib.machinery.SourceFileLoader('src.data_ingest', 'src/data_ingest.py')
di_spec = importlib.util.spec_from_loader(di_loader.name, di_loader)
di = importlib.util.module_from_spec(di_spec)
di.__package__ = 'src'
sys.modules['src.data_ingest'] = di
di_loader.exec_module(di)
src_pkg.data_ingest = di

loader = importlib.machinery.SourceFileLoader('src.research_ingest', 'src/research_ingest.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
ri = importlib.util.module_from_spec(spec)
ri.__package__ = 'src'
sys.modules['src.research_ingest'] = ri
src_pkg.research_ingest = ri
loader.exec_module(ri)


class TestResearchIngest(unittest.TestCase):
    def test_suggest_modules(self):
        papers = [{'title': 'Reinforcement Learning for Memory Models'}]
        mods = ri.suggest_modules(papers)
        self.assertIn('world_model_rl', mods)
        self.assertIn('hierarchical_memory', mods)


if __name__ == '__main__':
    unittest.main()
