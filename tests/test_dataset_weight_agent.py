import importlib.machinery
import importlib.util
import sys
import types
import tempfile
import sqlite3
from pathlib import Path
import unittest

src_pkg = types.ModuleType('src')
sys.modules['src'] = src_pkg

# load dependencies
for mod_name in ['license_inspector', 'dataset_bias_detector', 'fairness']:
    loader = importlib.machinery.SourceFileLoader(f'src.{mod_name}', f'src/{mod_name}.py')
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = 'src'
    sys.modules[f'src.{mod_name}'] = module
    loader.exec_module(module)
    setattr(src_pkg, mod_name, module)

loader = importlib.machinery.SourceFileLoader('src.dataset_weight_agent', 'src/dataset_weight_agent.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
agent_mod = importlib.util.module_from_spec(spec)
agent_mod.__package__ = 'src'
sys.modules['src.dataset_weight_agent'] = agent_mod
loader.exec_module(agent_mod)
DatasetWeightAgent = agent_mod.DatasetWeightAgent


class TestDatasetWeightAgent(unittest.TestCase):
    def test_sample_and_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / 'db.sqlite'
            conn = sqlite3.connect(db)
            conn.execute('CREATE TABLE datasets (name TEXT, source TEXT, url TEXT, license TEXT, license_text TEXT, weight REAL)')
            conn.execute("INSERT INTO datasets VALUES('ds1','src','u','MIT','','0')")
            conn.execute("INSERT INTO datasets VALUES('ds2','src','u2','Unknown','','0')")
            conn.close()

            agent = DatasetWeightAgent(db, allowed_licenses=['mit'], epsilon=0.0)
            sample = agent.sample(1)[0]
            self.assertEqual(sample, 'src:ds1')
            agent.observe('src:ds1', 1.0, {'g': {'tp': 1, 'fn': 0}})
            self.assertGreater(agent.weight('src:ds1'), 0.0)


if __name__ == '__main__':
    unittest.main()
