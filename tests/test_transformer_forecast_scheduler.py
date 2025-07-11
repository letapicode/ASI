import importlib.machinery
import importlib.util
import types
import sys
from unittest.mock import patch
import unittest

psutil_stub = types.SimpleNamespace(
    cpu_percent=lambda interval=None: 0.0,
    virtual_memory=lambda: types.SimpleNamespace(percent=0.0),
    net_io_counters=lambda: types.SimpleNamespace(bytes_sent=0, bytes_recv=0),
)
pynvml_stub = types.SimpleNamespace(
    nvmlInit=lambda: None,
    nvmlDeviceGetCount=lambda: 0,
    nvmlDeviceGetHandleByIndex=lambda i: i,
    nvmlDeviceGetPowerUsage=lambda h: 0,
)
sys.modules['psutil'] = psutil_stub
sys.modules['pynvml'] = pynvml_stub
if 'torch' in sys.modules:
    del sys.modules['torch']

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

hpc_mod = _load('asi.hpc_schedulers', 'src/hpc_schedulers.py')
mod = _load('asi.transformer_forecast_scheduler', 'src/transformer_forecast_scheduler.py')
TransformerForecastScheduler = mod.TransformerForecastScheduler


class TestTransformerForecastScheduler(unittest.TestCase):
    def test_predict_slot(self):
        sched = TransformerForecastScheduler(carbon_weight=1.0, cost_weight=1.0)
        with patch.object(sched, '_predict', return_value=([10, 1], [1.0, 0.2])):
            slot = sched.predict_slot(max_delay=7200.0)
            self.assertEqual(slot, 1)

    def test_submit_at_optimal_time(self):
        sched = TransformerForecastScheduler(carbon_weight=1.0, cost_weight=1.0)
        with patch.object(sched, '_predict', return_value=([10, 1], [1.0, 0.2])), \
             patch('time.sleep') as sl, \
             patch('subprocess.run') as sp:
            sp.return_value = types.SimpleNamespace(stdout='jid', returncode=0)
            jid = sched.submit_at_optimal_time(['run.sh'], max_delay=7200.0)
            sp.assert_called()
            self.assertEqual(jid, 'jid')


if __name__ == '__main__':
    unittest.main()
