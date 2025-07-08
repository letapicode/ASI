import importlib.machinery
import importlib.util
import sys
import types
import unittest
from unittest.mock import patch

psutil_stub = types.SimpleNamespace(sensors_battery=lambda: types.SimpleNamespace(percent=100))
sys.modules['psutil'] = psutil_stub

torch_stub = types.SimpleNamespace(
    cuda=types.SimpleNamespace(
        is_available=lambda: False,
        memory_allocated=lambda: 0,
        get_device_properties=lambda _: types.SimpleNamespace(total_memory=1),
    )
)
sys.modules['torch'] = torch_stub

src_pkg = types.ModuleType('src')
sys.modules['src'] = src_pkg
src_pkg.__path__ = ['src']
src_pkg.__spec__ = importlib.machinery.ModuleSpec('src', None, is_package=True)
pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg
pkg.__path__ = ['src']


def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'src'
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod

telemetry_mod = _load('asi.telemetry', 'src/telemetry.py')
TelemetryLogger = telemetry_mod.TelemetryLogger
sched_mod = _load('asi.adaptive_multi_objective_scheduler', 'src/adaptive_multi_objective_scheduler.py')
Scheduler = sched_mod.AdaptiveMultiObjectiveScheduler


class TestAdaptiveMultiObjectiveScheduler(unittest.TestCase):
    def test_submit_job(self):
        logger = TelemetryLogger(interval=0.01, carbon_data={'default': 1.0}, energy_price_data={'default': 1.0})
        history = [(1.0, 1.0, 1.0, 0.0), (0.5, 0.5, 1.0, 0.0)]
        sched = Scheduler(history, telemetry=logger, check_interval=0.01)
        with patch('asi.adaptive_multi_objective_scheduler.submit_job', return_value='jid') as sj, \
             patch.object(sched, '_policy', return_value=0):
            jid = sched.submit_job(['cmd'])
            self.assertEqual(jid, 'jid')
            sj.assert_called()


if __name__ == '__main__':
    unittest.main()
