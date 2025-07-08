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
src_pkg.__path__ = ['src']
src_pkg.__spec__ = importlib.machinery.ModuleSpec('src', None, is_package=True)
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
            entry = data[0]["outputs"][str(outp)]
            self.assertIn("hash", entry)
            self.assertIn("watermark_id", entry)
            steps = mgr.load()
            self.assertEqual(len(steps), 1)
            self.assertEqual(steps[0].note, "step1")

    def test_record_fairness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inp = root / "input.txt"
            outp = root / "output.txt"
            inp.write_text("in")
            outp.write_text("out")
            mgr = DatasetLineageManager(root)
            mgr.record([inp], [outp], note="step1")
            mgr.record([], [], note="fairness", fairness_before={"dp": 0.5}, fairness_after={"dp": 0.2})
            data = json.loads((root / "dataset_lineage.json").read_text())
            self.assertIn("fairness_before", data[1])
            self.assertEqual(data[1]["fairness_before"]["dp"], 0.5)


if __name__ == "__main__":
    unittest.main()

