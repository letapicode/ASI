import unittest
import tempfile
from pathlib import Path
import torch
from torch import nn
import importlib
import sys
import types
from pathlib import Path

asi_pkg = types.ModuleType('asi')
sys.modules.setdefault('asi', asi_pkg)
src_pkg = types.ModuleType('src')
src_pkg.__path__ = [str(Path('src'))]
sys.modules.setdefault('src', src_pkg)

DatasetVersioner = importlib.import_module('src.dataset_versioner').DatasetVersioner
ModelVersionManager = importlib.import_module('src.model_version_manager').ModelVersionManager


class TestModelVersionManager(unittest.TestCase):
    def test_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            dv = DatasetVersioner(tmp)
            f = Path(tmp) / "data.txt"
            f.write_text("hello")
            dv.record([f], note="t")
            model = nn.Linear(2, 1)
            mvm = ModelVersionManager(tmp, dv)
            h = mvm.record(model, epoch=1)
            log = Path(tmp) / "model_versions.json"
            self.assertTrue(log.exists())
            self.assertIn(h, log.read_text())


if __name__ == '__main__':
    unittest.main()
