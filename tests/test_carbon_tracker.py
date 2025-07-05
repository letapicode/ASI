import unittest
import time
import importlib.machinery
import importlib.util
import types
import sys

psutil_stub = types.SimpleNamespace(
    cpu_percent=lambda interval=None: 50.0,
    virtual_memory=lambda: types.SimpleNamespace(percent=10.0),
    net_io_counters=lambda: types.SimpleNamespace(bytes_sent=0, bytes_recv=0),
)
pynvml_stub = types.SimpleNamespace(
    nvmlInit=lambda: None,
    nvmlDeviceGetCount=lambda: 1,
    nvmlDeviceGetHandleByIndex=lambda i: i,
    nvmlDeviceGetPowerUsage=lambda h: 50000,
)
sys.modules['psutil'] = psutil_stub
sys.modules['pynvml'] = pynvml_stub

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

CarbonFootprintTracker = _load('asi.carbon_tracker', 'src/carbon_tracker.py').CarbonFootprintTracker
TelemetryLogger = _load('asi.telemetry', 'src/telemetry.py').TelemetryLogger
ComputeBudgetTracker = _load('asi.compute_budget_tracker', 'src/compute_budget_tracker.py').ComputeBudgetTracker


class TestCarbonTracker(unittest.TestCase):
    def test_energy(self):
        tracker = CarbonFootprintTracker(interval=0.1, co2_per_kwh=100.0)
        tracker.start()
        time.sleep(0.2)
        tracker.stop()
        stats = tracker.get_stats()
        self.assertGreater(stats['energy_kwh'], 0)
        self.assertAlmostEqual(stats['carbon_g'], stats['energy_kwh'] * 100.0)

    def test_telemetry_integration(self):
        cft = CarbonFootprintTracker(interval=0.1)
        logger = TelemetryLogger(interval=0.1, carbon_tracker=cft)
        logger.start()
        time.sleep(0.2)
        logger.stop()
        stats = logger.get_stats()
        self.assertIn('energy_kwh', stats)
        self.assertIn('carbon_g', stats)

    def test_budget_integration(self):
        cft = CarbonFootprintTracker(interval=0.1)
        logger = TelemetryLogger(interval=0.1, carbon_tracker=cft)
        tracker = ComputeBudgetTracker(0.001, telemetry=logger)
        tracker.start('run1')
        time.sleep(0.2)
        tracker.stop()
        usage = tracker.get_usage('run1')
        self.assertIn('carbon', usage)
        self.assertGreaterEqual(usage['carbon'], 0)

    def test_live_carbon_intensity(self):
        cft = CarbonFootprintTracker(interval=0.1, co2_per_kwh=50.0)
        logger = TelemetryLogger(interval=0.1, carbon_tracker=cft)
        logger.start()
        time.sleep(0.2)
        intensity = logger.get_live_carbon_intensity()
        logger.stop()
        self.assertAlmostEqual(intensity, 50.0, delta=1e-3)


if __name__ == '__main__':
    unittest.main()
