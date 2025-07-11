import importlib.machinery
import importlib.util
import types
import sys
from unittest.mock import patch
import unittest

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
requests_stub = types.ModuleType('requests')
requests_stub.get = lambda *a, **kw: types.SimpleNamespace(json=lambda: {}, raise_for_status=lambda: None)
sys.modules['requests'] = requests_stub

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

hpc_tel = _load('asi.telemetry', 'src/telemetry.py')
hpc_mod = _load('asi.hpc_schedulers', 'src/hpc_schedulers.py')
ct_mod = _load('asi.carbon_tracker', 'src/carbon_tracker.py')
carb_mod = _load('asi.carbon_hpc_scheduler', 'src/carbon_hpc_scheduler.py')
mod = _load('asi.cost_aware_scheduler', 'src/cost_aware_scheduler.py')
CarbonCostAwareScheduler = mod.CarbonCostAwareScheduler
get_hourly_price_forecast = mod.get_hourly_price_forecast


class TestCostAwareScheduler(unittest.TestCase):
    def test_get_hourly_price_forecast(self):
        resp = types.SimpleNamespace(
            json=lambda: {'forecast': [0.2, 0.1]},
            raise_for_status=lambda: None,
        )
        with patch('asi.cost_aware_scheduler.requests.get', return_value=resp):
            prices = get_hourly_price_forecast('aws', 'us', 'm5')
        self.assertEqual(prices, [0.2, 0.1])

    def test_combined_delay(self):
        sched = CarbonCostAwareScheduler(carbon_weight=1.0, cost_weight=1.0, threshold=0.5)
        with patch('asi.cost_aware_scheduler.get_hourly_forecast', return_value=[200, 50]), \
             patch('asi.cost_aware_scheduler.get_hourly_price_forecast', return_value=[1.0, 0.1]), \
             patch('time.sleep') as sl, \
             patch('asi.cost_aware_scheduler.submit_job', return_value='jid') as sj:
            jid = sched.submit_at_optimal_time(['run.sh'], max_delay=7200.0)
            sl.assert_called_with(3600.0)
            sj.assert_called_with(['run.sh'], backend='slurm')
            self.assertEqual(jid, 'jid')


if __name__ == '__main__':
    unittest.main()
