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

loader = importlib.machinery.SourceFileLoader('asi.hpc_schedulers', 'src/hpc_schedulers.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
mod_sched = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = mod_sched
loader.exec_module(mod_sched)

loader = importlib.machinery.SourceFileLoader('asi.carbon_tracker', 'src/carbon_tracker.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
mod_ct = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = mod_ct
loader.exec_module(mod_ct)

loader = importlib.machinery.SourceFileLoader('asi.carbon_aware_scheduler', 'src/carbon_aware_scheduler.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
mod = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = mod
loader.exec_module(mod)
CarbonAwareScheduler = mod.CarbonAwareScheduler
get_hourly_forecast = mod.get_hourly_forecast

class TestCarbonAwareScheduler(unittest.TestCase):
    def test_submit_when_green(self):
        sch = CarbonAwareScheduler(threshold=100.0, backend='slurm', carbon_api='u')
        resp = types.SimpleNamespace(
            json=lambda: {'data': [{'intensity': {'forecast': 50}}]},
            raise_for_status=lambda: None,
        )
        with patch('urllib.request.urlopen', return_value=types.SimpleNamespace(__enter__=lambda s: resp, __exit__=lambda *a: None)) as get, \
             patch('asi.hpc_schedulers.subprocess.run', return_value=types.SimpleNamespace(stdout='42', returncode=0, check_returncode=lambda: None)):
            job_id = sch.submit_when_green(['run.sh'])
            get.assert_called()
            self.assertEqual(job_id, '42')

    def test_get_hourly_forecast(self):
        resp = types.SimpleNamespace(
            json=lambda: {'data': [
                {'intensity': {'forecast': 10}},
                {'intensity': {'forecast': 20}},
            ]},
            raise_for_status=lambda: None,
        )
        requests_stub.get = lambda *a, **kw: resp
        hours = get_hourly_forecast()
        self.assertEqual(len(hours), 2)

if __name__ == '__main__':
    unittest.main()
