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
AdaptiveScheduler = _load('asi.adaptive_scheduler', 'src/adaptive_scheduler.py').AdaptiveScheduler


class TestAdaptiveScheduler(unittest.TestCase):
    def test_budget_pause(self):
        logger = TelemetryLogger(interval=0.05)
        tracker = ComputeBudgetTracker(0.0, telemetry=logger)
        sched = AdaptiveScheduler(tracker, 'run', check_interval=0.05)
        ran = []

        def job():
            ran.append(1)
            return 1.0

        sched.add(job)
        time.sleep(0.2)
        sched.stop()
        self.assertEqual(len(ran), 0)

    def test_improvement_pause(self):
        logger = TelemetryLogger(interval=0.05)
        tracker = ComputeBudgetTracker(1.0, telemetry=logger)
        sched = AdaptiveScheduler(tracker, 'run', check_interval=0.05, window=2, min_improvement=0.5)
        ran = []

        def job():
            ran.append(1)
            return 0.0

        sched.add(job)
        sched.add(job)
        sched.add(job)
        time.sleep(0.4)
        sched.stop()
        self.assertLessEqual(len(ran), 2)

    def test_report_load(self):
        logger = TelemetryLogger(interval=0.05)
        tracker = ComputeBudgetTracker(1.0, telemetry=logger)
        sched = AdaptiveScheduler(tracker, 'run', check_interval=0.05)
        load = sched.report_load()
        self.assertIsInstance(load, dict)
        self.assertIn('cpu', load)
        sched.stop()

    def test_carbon_priority(self):
        logger = TelemetryLogger(interval=0.05, carbon_data={'US': 0.5, 'EU': 0.3})
        tracker = ComputeBudgetTracker(1.0, telemetry=logger)
        sched = AdaptiveScheduler(tracker, 'run', check_interval=0.05)
        order: list[str] = []

        def job_a():
            order.append('a')

        def job_b():
            order.append('b')

        sched.add(job_a, region='US')
        sched.add(job_b, region='EU')
        time.sleep(0.3)
        sched.stop()
        self.assertTrue(order and order[0] == 'b')


if __name__ == '__main__':
    unittest.main()
