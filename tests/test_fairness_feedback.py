import importlib.machinery
import importlib.util
import sys
import types
import tempfile
import sqlite3
import json
from pathlib import Path
import unittest

src_pkg = types.ModuleType('src')
sys.modules['src'] = src_pkg
src_pkg.__path__ = ['src']
src_pkg.__spec__ = importlib.machinery.ModuleSpec('src', None, is_package=True)
import numpy as np
torch_stub = types.SimpleNamespace(
    Tensor=np.ndarray,
    float32=float,
    nn=types.SimpleNamespace(Module=object),
    zeros=lambda n, dtype=None: np.zeros(n, dtype=float),
    softmax=lambda x, dim=0: np.exp(x) / np.exp(x).sum(),
)
sys.modules.setdefault('torch', torch_stub)
sys.modules.setdefault('requests', types.ModuleType('requests'))
sys.modules.setdefault('PIL', types.ModuleType('PIL'))
sys.modules.setdefault('PIL.Image', types.ModuleType('PIL.Image'))
sys.modules.setdefault('PIL.PngImagePlugin', types.ModuleType('PIL.PngImagePlugin'))
sys.modules.setdefault('PIL.UnidentifiedImageError', type('E', (), {}))
robot_skill_stub = types.ModuleType('src.robot_skill_transfer')
robot_skill_stub.VideoPolicyDataset = list
sys.modules.setdefault('src.robot_skill_transfer', robot_skill_stub)


def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(f'src.{name}', path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'src'
    sys.modules[f'src.{name}'] = mod
    loader.exec_module(mod)
    setattr(src_pkg, name.split('.')[-1], mod)
    return mod

# load dependencies

_load('fairness_evaluator', 'src/fairness_evaluator.py')

data_ingest = _load('data_ingest', 'src/data_ingest.py')
_load('dataset_lineage_manager', 'src/dataset_lineage_manager.py')
_load('adaptive_curriculum', 'src/adaptive_curriculum.py')
_load('cross_lingual_fairness', 'src/cross_lingual_fairness.py')
_load('dataset_weight_agent', 'src/dataset_weight_agent.py')
ff_mod = _load('fairness_feedback', 'src/fairness_feedback.py')

ActiveDataSelector = data_ingest.ActiveDataSelector
DatasetLineageManager = sys.modules['src.dataset_lineage_manager'].DatasetLineageManager
WeightAgent = sys.modules['src.dataset_weight_agent'].DatasetWeightAgent
FairnessFeedback = ff_mod.FairnessFeedback


class TestFairnessFeedback(unittest.TestCase):
    def test_update_and_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / 'db.sqlite'
            conn = sqlite3.connect(db)
            conn.execute('CREATE TABLE datasets (name TEXT, source TEXT, url TEXT, license TEXT, license_text TEXT, weight REAL)')
            conn.execute("INSERT INTO datasets VALUES('ds','src','u','MIT','','0')")
            conn.commit()
            conn.close()

            lineage = DatasetLineageManager(tmp)
            selector = ActiveDataSelector(threshold=1.0)
            agent = WeightAgent(db)
            fb = FairnessFeedback(selector, agent, gap_threshold=0.0, lineage=lineage)

            stats = {'hola': {'tp': 1, 'fn': 0}, '[en] hola': {'tp': 0, 'fn': 1}}
            metrics = fb.update(stats, dataset='src:ds', val_accuracy=1.0)
            self.assertIn('demographic_parity', metrics)
            self.assertNotEqual(selector.threshold, 1.0)

            with sqlite3.connect(db) as c:
                weight = c.execute('SELECT weight FROM datasets').fetchone()[0]
            self.assertNotEqual(weight, 0.0)

            log = json.loads((Path(tmp) / 'dataset_lineage.json').read_text())
            self.assertEqual(len(log), 1)
            self.assertIn('fairness_feedback', log[0]['note'])
            self.assertIn('metrics', log[0]['note'])


if __name__ == '__main__':
    unittest.main()
