import json
import tempfile
import unittest
from pathlib import Path
import importlib.machinery
import importlib.util
import types
import sys

src_pkg = types.ModuleType('src')
src_pkg.__path__ = ['src']
src_pkg.__spec__ = importlib.machinery.ModuleSpec('src', None, is_package=True)
sys.modules['src'] = src_pkg
loader = importlib.machinery.SourceFileLoader('src.license_inspector', 'src/license_inspector.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
li = importlib.util.module_from_spec(spec)
li.__package__ = 'src'
sys.modules['src.license_inspector'] = li
loader.exec_module(li)
LicenseInspector = li.LicenseInspector

loader2 = importlib.machinery.SourceFileLoader('src.dataset_discovery', 'src/dataset_discovery.py')
spec2 = importlib.util.spec_from_loader(loader2.name, loader2)
dd = importlib.util.module_from_spec(spec2)
dd.__package__ = 'src'
sys.modules['src.dataset_discovery'] = dd
loader2.exec_module(dd)
DiscoveredDataset = dd.DiscoveredDataset
store_datasets = dd.store_datasets

loader3 = importlib.machinery.SourceFileLoader('src.dataset_lineage', 'src/dataset_lineage.py')
spec3 = importlib.util.spec_from_loader(loader3.name, loader3)
dlm = importlib.util.module_from_spec(spec3)
dlm.__package__ = 'src'
sys.modules['src.dataset_lineage'] = dlm
loader3.exec_module(dlm)
DatasetLineageManager = dlm.DatasetLineageManager


class TestLicenseInspector(unittest.TestCase):
    def test_inspect(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = Path(tmp) / 'sample.json'
            meta.write_text(json.dumps({'license': 'MIT'}))
            insp = LicenseInspector()
            res = insp.inspect_dir(tmp)
            self.assertTrue(res[str(meta)])

    def test_scan_discovered(self):
        rss = """<rss><channel>
        <item><title>ds1</title><link>http://x/ds1</link><license>MIT</license></item>
        <item><title>ds2</title><link>http://x/ds2</link><license>GPL</license></item>
        </channel></rss>"""
        dsets = dd._parse_rss(rss, 'hf')
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / 'db.sqlite'
            store_datasets(dsets, db)
            lin_root = Path(tmp) / 'lineage'
            lin_root.mkdir()
            lineage = DatasetLineageManager(lin_root)
            insp = LicenseInspector(['mit'])
            res = insp.scan_discovered_db(db, lineage)
            self.assertTrue(res['hf:ds1'])
            self.assertFalse(res['hf:ds2'])
            data = json.loads((lin_root / 'dataset_lineage.json').read_text())
            self.assertEqual(len(data), 2)


if __name__ == '__main__':
    unittest.main()
