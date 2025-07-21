import unittest
import time
import importlib.machinery
import importlib.util
import importlib
import types
import os
import sys

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

ct_stub = types.ModuleType('asi.carbon_tracker')
class _CT:
    def __init__(self, *a, **k):
        pass
    def start(self):
        pass
    def stop(self):
        pass
    def get_stats(self):
        return {}
ct_stub.CarbonFootprintTracker = _CT
sys.modules['asi.carbon_tracker'] = ct_stub
med_stub = types.ModuleType('asi.memory_event_detector')
class _MED:
    def __init__(self, *a, **k):
        self.events = []
    def update(self, snapshot):
        return []
med_stub.MemoryEventDetector = _MED
sys.modules['asi.memory_event_detector'] = med_stub
cas_stub = types.ModuleType('asi.cost_aware_scheduler')
cas_stub.get_current_price = lambda *a, **k: 0.0
sys.modules['asi.cost_aware_scheduler'] = cas_stub

sys.modules['asi.fpga_backend'] = types.SimpleNamespace(_HAS_FPGA=False, cl=None)
sys.modules['asi.analog_backend'] = types.SimpleNamespace(
    _HAS_ANALOG=True,
    analogsim=types.SimpleNamespace(utilization=lambda: 0.0)
)
sys.modules['asi.loihi_backend'] = types.SimpleNamespace(_HAS_LOIHI=False)


def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = name.rpartition('.')[0]
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod

TelemetryLogger = _load('asi.telemetry', 'src/telemetry.py').TelemetryLogger
sys.modules['asi.telemetry']._HAS_PROM = False
sys.modules['asi.telemetry'].torch = torch
hardware_detect = _load('asi.hardware_detect', 'src/hardware_detect.py')
hardware_detect.list_cpus = lambda: ['cpu0']
hardware_detect.list_gpus = lambda: ['gpu0']
hardware_detect.list_fpgas = lambda: ['fpga0']
hardware_detect.list_loihi = lambda: []
hardware_detect.list_analog = hardware_detect.list_analog
ComputeBudgetTracker = _load('asi.compute_budget_tracker', 'src/compute_budget_tracker.py').ComputeBudgetTracker
AdaptiveScheduler = _load('asi.schedulers', 'src/schedulers.py').AdaptiveScheduler
sys.modules['asi.schedulers'].torch = torch


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
        time.sleep(0.1)
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

    def test_device_queueing(self):
        mod = sys.modules['asi.schedulers']
        sys.modules['torch'] = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                is_available=lambda: True,
                memory_allocated=lambda: 0,
                get_device_properties=lambda idx: types.SimpleNamespace(total_memory=1),
                utilization=lambda: 0.0,
            )
        )
        mod.torch = sys.modules['torch']
        if hasattr(mod, 'psutil'):
            mod.psutil = types.SimpleNamespace(cpu_percent=lambda interval=None: 0.0)

        os.environ['ASI_ANALOG_DEVICES'] = 'analog0'
        try:
            logger = TelemetryLogger(interval=0.05)
            tracker = ComputeBudgetTracker(1.0, telemetry=logger)
            sched = AdaptiveScheduler(tracker, 'run', check_interval=0.05)
            ran: list[str] = []

            sched.add(lambda: ran.append('cpu'), device='cpu')
            sched.add(lambda: ran.append('gpu'), device='gpu')
            sched.add(lambda: ran.append('fpga'), device='fpga')
            sched.add(lambda: ran.append('analog'), device='analog')

            time.sleep(0.3)
            sched.stop()
            self.assertEqual(set(ran), {'cpu', 'gpu', 'fpga', 'analog'})
        finally:
            os.environ.pop('ASI_ANALOG_DEVICES', None)


if __name__ == '__main__':
    unittest.main()
