import sqlite3
import tempfile
import unittest
from pathlib import Path
import importlib.machinery
import importlib.util
import sys
import types

src_pkg = types.ModuleType('src')
sys.modules['src'] = src_pkg

loader = importlib.machinery.SourceFileLoader('src.dataset_discovery', 'src/dataset_discovery.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
dd = importlib.util.module_from_spec(spec)
dd.__package__ = 'src'
sys.modules['src.dataset_discovery'] = dd
loader.exec_module(dd)

loader2 = importlib.machinery.SourceFileLoader('src.rl_dataset_discovery', 'src/rl_dataset_discovery.py')
spec2 = importlib.util.spec_from_loader(loader2.name, loader2)
rl = importlib.util.module_from_spec(spec2)
rl.__package__ = 'src'
sys.modules['src.rl_dataset_discovery'] = rl
loader2.exec_module(rl)

DiscoveredDataset = dd.DiscoveredDataset
store_datasets = dd.store_datasets
DatasetQualityAgent = rl.DatasetQualityAgent


class TestRLDatasetDiscovery(unittest.TestCase):
    def test_weight_assignment(self):
        rss = """<rss><channel>
        <item><title>unique ds</title><link>http://x/ds1</link><license>MIT</license></item>
        <item><title>dup dup</title><link>http://x/ds2</link><license>Unknown</license></item>
        </channel></rss>"""
        dsets = dd._parse_rss(rss, 'hf')
        agent = DatasetQualityAgent(['mit'])
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / 'db.sqlite'
            store_datasets(dsets, db, agent=agent)
            conn = sqlite3.connect(db)
            cur = conn.execute('SELECT weight FROM datasets ORDER BY name')
            w1, w2 = [r[0] for r in cur.fetchall()]
            conn.close()
            self.assertGreater(w1, w2)


if __name__ == '__main__':
    unittest.main()
