import sqlite3
import tempfile
import unittest
from pathlib import Path

import importlib.machinery
import importlib.util
import types

src_pkg = types.ModuleType('src')
src_pkg.__path__ = ['src']
src_pkg.__spec__ = importlib.machinery.ModuleSpec('src', None, is_package=True)
import sys
sys.modules['src'] = src_pkg
loader = importlib.machinery.SourceFileLoader('src.dataset_discovery', 'src/dataset_discovery.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
dd = importlib.util.module_from_spec(spec)
dd.__package__ = 'src'
sys.modules['src.dataset_discovery'] = dd
loader.exec_module(dd)
DiscoveredDataset = dd.DiscoveredDataset
parse_hf = dd.discover_huggingface
store_datasets = dd.store_datasets

loader_li = importlib.machinery.SourceFileLoader('src.license_inspector', 'src/license_inspector.py')
spec_li = importlib.util.spec_from_loader(loader_li.name, loader_li)
li = importlib.util.module_from_spec(spec_li)
li.__package__ = 'src'
sys.modules['src.license_inspector'] = li
loader_li.exec_module(li)
LicenseInspector = li.LicenseInspector


class TestDatasetDiscovery(unittest.TestCase):
    def test_store_and_inspect(self):
        rss = """<rss><channel>
        <item><title>ds1</title><link>http://x/ds1</link><license>MIT</license></item>
        <item><title>ds2</title><link>http://x/ds2</link><license>Apache</license></item>
        </channel></rss>"""
        dsets = dd._parse_rss(rss, 'hf')
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / 'db.sqlite'
            store_datasets(dsets, db)
            conn = sqlite3.connect(db)
            cur = conn.execute('select count(*) from datasets')
            self.assertEqual(cur.fetchone()[0], 2)
            conn.close()
            insp = LicenseInspector(['mit'])
            res = insp.inspect_db(db)
            self.assertTrue(res['hf:ds1'])
            self.assertFalse(res['hf:ds2'])


if __name__ == '__main__':
    unittest.main()
