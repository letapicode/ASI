import json
import os
import tempfile
import unittest
from pathlib import Path
import sys

import importlib.machinery
import importlib.util
import types

src_pkg = types.ModuleType('src')
sys.modules['src'] = src_pkg
loader = importlib.machinery.SourceFileLoader('src.dataset_lineage_manager', 'src/dataset_lineage_manager.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
dlm = importlib.util.module_from_spec(spec)
dlm.__package__ = 'src'
sys.modules['src.dataset_lineage_manager'] = dlm
loader.exec_module(dlm)
DatasetLineageManager = dlm.DatasetLineageManager


class TestDatasetLineageManager(unittest.TestCase):
    def test_record_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inp = root / "input.txt"
            outp = root / "output.txt"
            inp.write_text("in")
            outp.write_text("out")
            mgr = DatasetLineageManager(root)
            mgr.record([inp], [outp], note="step1")
            data = json.loads((root / "dataset_lineage.json").read_text())
            self.assertEqual(data[0]["note"], "step1")
            self.assertEqual(data[0]["inputs"], [str(inp)])
            self.assertIn(str(outp), data[0]["outputs"])
            steps = mgr.load()
            self.assertEqual(len(steps), 1)
            self.assertEqual(steps[0].note, "step1")


if __name__ == "__main__":
    unittest.main()

