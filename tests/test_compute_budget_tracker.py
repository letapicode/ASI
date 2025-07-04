import unittest
import time
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
    mod.__package__ = name.rpartition('.')[0]
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod

TelemetryLogger = _load('asi.telemetry', 'src/telemetry.py').TelemetryLogger
ComputeBudgetTracker = _load('asi.compute_budget_tracker', 'src/compute_budget_tracker.py').ComputeBudgetTracker


class TestComputeBudgetTracker(unittest.TestCase):
    def test_tracking(self):
        logger = TelemetryLogger(interval=0.1)
        tracker = ComputeBudgetTracker(0.001, telemetry=logger)
        tracker.start('run1')
        time.sleep(0.2)
        tracker.stop()
        usage = tracker.get_usage('run1')
        self.assertIn('gpu_hours', usage)
        self.assertIn('mem_peak', usage)
        self.assertIn('carbon', usage)
        self.assertLessEqual(tracker.remaining('run1'), 0.001)


if __name__ == '__main__':
    unittest.main()
