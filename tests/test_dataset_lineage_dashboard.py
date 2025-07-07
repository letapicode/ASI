import unittest
import tempfile
import json
import http.client
from pathlib import Path
import importlib.machinery
import importlib.util
import sys
import types

src_pkg = types.ModuleType('src')
sys.modules['src'] = src_pkg
src_pkg.__path__ = ['src']
src_pkg.__spec__ = importlib.machinery.ModuleSpec('src', None, is_package=True)


def _load(name: str, path: str):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'src'
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod

DatasetLineageManager = _load('src.dataset_lineage_manager', 'src/dataset_lineage_manager.py').DatasetLineageManager
DatasetLineageDashboard = _load('src.dataset_lineage_dashboard', 'src/dataset_lineage_dashboard.py').DatasetLineageDashboard


class TestDatasetLineageDashboard(unittest.TestCase):
    def test_http_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inp = root / 'in.txt'
            outp = root / 'out.txt'
            inp.write_text('a')
            outp.write_text('b')
            mgr = DatasetLineageManager(root)
            mgr.record([inp], [outp], note='copy')
            dash = DatasetLineageDashboard(mgr)
            dash.start(port=0)
            port = dash.port
            conn = http.client.HTTPConnection('localhost', port)
            conn.request('GET', '/graph')
            resp = conn.getresponse()
            data = json.loads(resp.read())
            self.assertEqual(len(data['nodes']), 2)
            conn.request('GET', '/steps?q=copy')
            resp = conn.getresponse()
            steps = json.loads(resp.read())
            self.assertEqual(len(steps), 1)
            dash.stop()


if __name__ == '__main__':
    unittest.main()
