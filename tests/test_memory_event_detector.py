import unittest
import importlib.machinery
import importlib.util
import types
import sys

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg

def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    sys.modules[name] = mod
    return mod

mod = _load('asi.memory_event_detector', 'src/memory_event_detector.py')
detect_change_points = mod.detect_change_points
MemoryEventDetector = mod.MemoryEventDetector

class TestMemoryEventDetector(unittest.TestCase):
    def test_detect_change_points(self):
        series = [1.0] * 5 + [5.0] * 3
        cps = detect_change_points(series, window=3, threshold=1.5)
        self.assertIn(5, cps)

    def test_parse_and_update(self):
        lines = ['{"cpu": 1}', '{"cpu": 1}', '{"cpu": 5}']
        entries = MemoryEventDetector.parse_log(lines)
        det = MemoryEventDetector(window=2, threshold=1.0)
        for e in entries:
            det.update(e)
        self.assertTrue(det.events)
        self.assertEqual(det.events[0]['metric'], 'cpu')

if __name__ == '__main__':
    unittest.main()
