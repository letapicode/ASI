import importlib.machinery
import importlib.util
import types
import sys
import unittest

psutil_stub = types.SimpleNamespace(
    cpu_percent=lambda interval=None: 0.0,
    virtual_memory=lambda: types.SimpleNamespace(percent=0.0),
    net_io_counters=lambda: types.SimpleNamespace(bytes_sent=0, bytes_recv=0),
)
psutil_stub.sensors_battery = lambda: None
pynvml_stub = types.SimpleNamespace(
    nvmlInit=lambda: None,
    nvmlDeviceGetCount=lambda: 0,
    nvmlDeviceGetHandleByIndex=lambda i: i,
    nvmlDeviceGetPowerUsage=lambda h: 0,
)
sys.modules['psutil'] = psutil_stub
sys.modules['pynvml'] = pynvml_stub

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg
pkg.__path__ = ['src']


def _load(name: str, path: str):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = name.rpartition('.')[0]
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod


telemetry_mod = _load('asi.telemetry', 'src/telemetry.py')
TelemetryLogger = telemetry_mod.TelemetryLogger


class TestTelemetryRolling(unittest.TestCase):
    def test_rolling_metrics(self) -> None:
        logger = TelemetryLogger(interval=0.05)
        logger.history = [
            {'carbon_g': 1.0, 'energy_cost': 2.0},
            {'carbon_g': 3.0, 'energy_cost': 4.0},
        ]
        res = logger.rolling_metrics(window=1)
        self.assertEqual(res['carbon_g'], 3.0)
        self.assertEqual(res['energy_cost'], 4.0)
        res = logger.rolling_metrics(window=2)
        self.assertAlmostEqual(res['carbon_g'], 2.0)
        self.assertAlmostEqual(res['energy_cost'], 3.0)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()

