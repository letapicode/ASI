import unittest
import importlib.machinery
import importlib.util
import sys

loader = importlib.machinery.SourceFileLoader('src.data_poison_detector', 'src/data_poison_detector.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
mod = importlib.util.module_from_spec(spec)
mod.__package__ = 'src'
sys.modules['src.data_poison_detector'] = mod
loader.exec_module(mod)
DataPoisonDetector = mod.DataPoisonDetector


class TestDataPoisonDetector(unittest.TestCase):
    def test_detect_poison(self):
        det = DataPoisonDetector(window=3, clusters=2, threshold=2.0)
        self.assertFalse(det.record_text("hello world"))
        self.assertFalse(det.record_text("goodbye world"))
        poison = " ".join(f"w{i}" for i in range(20))
        self.assertTrue(det.record_text(poison))


if __name__ == "__main__":
    unittest.main()
