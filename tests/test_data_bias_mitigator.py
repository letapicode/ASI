import unittest
import tempfile
from pathlib import Path
import importlib.machinery
import importlib.util
import types
import sys

src_pkg = types.ModuleType('src')
src_pkg.__path__ = ['src']
src_pkg.__spec__ = importlib.machinery.ModuleSpec('src', None, is_package=True)
sys.modules['src'] = src_pkg
loader = importlib.machinery.SourceFileLoader('src.data_bias_mitigator', 'src/data_bias_mitigator.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
mod = importlib.util.module_from_spec(spec)
mod.__package__ = 'src'
sys.modules['src.data_bias_mitigator'] = mod
loader.exec_module(mod)
DataBiasMitigator = mod.DataBiasMitigator


class TestDataBiasMitigator(unittest.TestCase):
    def test_reweight_and_filter(self):
        mit = DataBiasMitigator(threshold=0.5)
        with tempfile.TemporaryDirectory() as d:
            p1 = Path(d) / "high.txt"
            p2 = Path(d) / "low.txt"
            p1.write_text("hello hello hello")
            p2.write_text("alpha beta gamma delta")
            weights = mit.reweight_files([p1, p2])
            self.assertLess(weights[p1], weights[p2])
            kept = mit.filter_files([p1, p2])
            self.assertEqual(kept, [p2])


if __name__ == '__main__':
    unittest.main()
