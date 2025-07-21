import importlib.util
import types
import sys
import time
import unittest

# stub psutil and torch
psutil_stub = types.SimpleNamespace(
    sensors_battery=lambda: types.SimpleNamespace(percent=10, power_plugged=False),
    cpu_percent=lambda interval=None: 0.0,
    cpu_count=lambda logical=False: 1,
    virtual_memory=lambda: types.SimpleNamespace(percent=0.0),
    net_io_counters=lambda: types.SimpleNamespace(bytes_sent=0, bytes_recv=0),
)
torch_stub = types.SimpleNamespace(
    cuda=types.SimpleNamespace(
        utilization=lambda: 0.0,
        is_available=lambda: False,
        memory_allocated=lambda: 0,
        get_device_properties=lambda _: types.SimpleNamespace(total_memory=1),
    )
)
sys.modules['psutil'] = psutil_stub
sys.modules['torch'] = torch_stub

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg
pkg.__path__ = ['src']


def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = name.rpartition('.')[0]
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod

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
cas_stub.get_current_price = lambda provider, region, inst: 0.0
sys.modules['asi.cost_aware_scheduler'] = cas_stub
TelemetryLogger = _load('asi.telemetry', 'src/telemetry.py').TelemetryLogger
sys.modules['asi.telemetry']._HAS_PROM = False
ComputeBudgetTracker = _load('asi.compute_budget_tracker', 'src/compute_budget_tracker.py').ComputeBudgetTracker
AdaptiveScheduler = _load('asi.schedulers', 'src/schedulers.py').AdaptiveScheduler
BatteryAwareScheduler = _load('asi.schedulers', 'src/schedulers.py').BatteryAwareScheduler


class TestBatteryAwareScheduler(unittest.TestCase):
    def test_pause_low_battery(self):
        logger = TelemetryLogger(interval=0.05)
        tracker = ComputeBudgetTracker(1.0, telemetry=logger)
        sched = BatteryAwareScheduler(tracker, 'run', check_interval=0.05, battery_threshold=0.5)
        ran = []

        def job():
            ran.append(1)

        sched.add(job)
        time.sleep(0.2)
        sched.stop()
        self.assertLess(logger.metrics.get('battery', 100.0), 100.0)

    def test_adaptive_integration(self):
        psutil_stub.sensors_battery = lambda: types.SimpleNamespace(percent=80)
        logger = TelemetryLogger(interval=0.05)
        tracker = ComputeBudgetTracker(1.0, telemetry=logger)
        sched = AdaptiveScheduler(tracker, 'run', check_interval=0.05, battery_scheduler=True, battery_threshold=0.5)
        self.assertIsInstance(sched, BatteryAwareScheduler)
        ran = []

        def job():
            ran.append(1)

        sched.add(job)
        time.sleep(0.2)
        sched.stop()
        self.assertGreaterEqual(len(ran), 1)


if __name__ == '__main__':
    unittest.main()
