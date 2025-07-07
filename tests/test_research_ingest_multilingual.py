import unittest
import importlib.machinery
import importlib.util
import types
import sys
import tempfile
import json
from pathlib import Path
from unittest.mock import patch

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

# load data_ingest and research_ingest dynamically
loader_di = importlib.machinery.SourceFileLoader('src.data_ingest', 'src/data_ingest.py')
spec_di = importlib.util.spec_from_loader(loader_di.name, loader_di)
di = importlib.util.module_from_spec(spec_di)
di.__package__ = 'src'
loader_di.exec_module(di)
sys.modules['src.data_ingest'] = di
src_pkg.data_ingest = di
CrossLingualTranslator = di.CrossLingualTranslator

loader_ri = importlib.machinery.SourceFileLoader('src.research_ingest', 'src/research_ingest.py')
spec_ri = importlib.util.spec_from_loader(loader_ri.name, loader_ri)
ri = importlib.util.module_from_spec(spec_ri)
ri.__package__ = 'src'
loader_ri.exec_module(ri)
sys.modules['src.research_ingest'] = ri
src_pkg.research_ingest = ri


class TestResearchIngestMultilingual(unittest.TestCase):
    def test_run_ingestion_translates(self):
        tr = CrossLingualTranslator(['es'])
        with tempfile.TemporaryDirectory() as root:
            with patch.object(ri, 'fetch_recent_papers', return_value=[{'title': 'T', 'summary': 'S'}]):
                path = ri.run_ingestion(root, translator=tr)
            data = json.loads(Path(path).read_text())
            paper = data['papers'][0]
            self.assertEqual(paper['title_translations']['es'], '[es] T')
            self.assertEqual(paper['summary_translations']['es'], '[es] S')


if __name__ == '__main__':
    unittest.main()
