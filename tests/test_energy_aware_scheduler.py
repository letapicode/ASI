import importlib.machinery
import importlib.util
import types
import sys
import time
import unittest

# stub torch
torch_stub = types.SimpleNamespace(
    cuda=types.SimpleNamespace(
        utilization=lambda: 0.0,
        is_available=lambda: False,
        memory_allocated=lambda: 0,
        get_device_properties=lambda _: types.SimpleNamespace(total_memory=1),
    )
)
torch_stub.nn = types.SimpleNamespace(Module=type('M', (), {}))
sys.modules['torch'] = torch_stub

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg
sys.modules['asi.fpga_backend'] = types.SimpleNamespace(_HAS_FPGA=False, cl=None)
sys.modules['asi.analog_backend'] = types.SimpleNamespace(_HAS_ANALOG=False)
sys.modules['asi.loihi_backend'] = types.SimpleNamespace(_HAS_LOIHI=False)
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


def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = name.rpartition('.')[0]
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod

TelemetryLogger = _load('asi.telemetry', 'src/telemetry.py').TelemetryLogger
hardware_detect = _load('asi.hardware_detect', 'src/hardware_detect.py')
hardware_detect.list_cpus = lambda: ['cpu0']
hardware_detect.list_gpus = lambda: ['gpu0']
hardware_detect.list_fpgas = lambda: []
hardware_detect.list_loihi = lambda: []
hardware_detect.list_analog = lambda: []
ComputeBudgetTracker = _load('asi.compute_budget_tracker', 'src/compute_budget_tracker.py').ComputeBudgetTracker
AdaptiveScheduler = _load('asi.adaptive_scheduler', 'src/adaptive_scheduler.py').AdaptiveScheduler
EnergyAwareScheduler = _load('asi.energy_aware_scheduler', 'src/energy_aware_scheduler.py').EnergyAwareScheduler


class TestEnergyAwareScheduler(unittest.TestCase):
    def test_delay(self):
        logger = TelemetryLogger(interval=0.05, carbon_data={'default': 1.0})
        tracker = ComputeBudgetTracker(1.0, telemetry=logger)
        sched = EnergyAwareScheduler(tracker, 'run', check_interval=0.05, intensity_threshold=0.5)
        ran = []

        def job():
            ran.append(1)

        sched.add(job)
        time.sleep(0.2)
        sched.stop()
        self.assertEqual(len(ran), 0)

    def test_migrate_via_adaptive(self):
        logger = TelemetryLogger(interval=0.05, carbon_data={'US': 1.0, 'EU': 0.1})
        tracker = ComputeBudgetTracker(1.0, telemetry=logger)
        sched = AdaptiveScheduler(tracker, 'run', check_interval=0.05, energy_scheduler=True, intensity_threshold=0.5)
        self.assertIsInstance(sched, EnergyAwareScheduler)
        ran = []

        def job():
            ran.append(1)

        sched.add(job, region='US')
        time.sleep(0.3)
        sched.stop()
        self.assertEqual(len(ran), 1)


if __name__ == '__main__':
    unittest.main()
