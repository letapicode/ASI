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
sys.modules['numpy'] = types.ModuleType('numpy')
sys.modules['statsmodels'] = types.ModuleType('statsmodels')
sys.modules['statsmodels.tsa'] = types.ModuleType('statsmodels.tsa')
sys.modules['statsmodels.tsa.arima'] = types.ModuleType('statsmodels.tsa.arima')
sys.modules['statsmodels.tsa.arima.model'] = types.ModuleType('statsmodels.tsa.arima.model')
sys.modules['statsmodels.tsa.arima.model'].ARIMA = object

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

base_mod = _load('asi.hpc_base_scheduler', 'src/hpc_base_scheduler.py')
deep_mod = _load('asi.rl_schedulers', 'src/rl_schedulers.py')
make_scheduler = base_mod.make_scheduler
DeepRLScheduler = deep_mod.DeepRLScheduler


class TestDeepRLScheduler(unittest.TestCase):
    def test_schedule_job(self):
        a = make_scheduler('arima')
        b = make_scheduler('arima', backend='k8s')
        sched = DeepRLScheduler({'a': a, 'b': b})
        with patch.object(sched, '_predict', side_effect=[([10, 1], [1.0, 0.2]), ([5, 0.5], [0.5, 0.1])]), \
             patch('time.sleep') as sl, \
             patch('asi.rl_schedulers.submit_job', return_value='jid') as sj:
            cluster, jid = sched.schedule_job(['run.sh'], max_delay=7200.0)
            sl.assert_called_with(3600.0)
            sj.assert_called_with(['run.sh'], backend='k8s')
            self.assertEqual(cluster, 'b')
            self.assertEqual(jid, 'jid')

    def test_fallback_without_torch(self):
        a = make_scheduler('arima', carbon_history=[1.0], cost_history=[0.5])
        deep_mod.torch = None
        sched = DeepRLScheduler({'a': a})
        with patch('asi.rl_schedulers.submit_job', return_value='jid') as sj:
            cluster, jid = sched.schedule_job(['run.sh'], max_delay=0.0)
            self.assertEqual(cluster, 'a')
            self.assertEqual(jid, 'jid')
            sj.assert_called_with(['run.sh'], backend='slurm')


if __name__ == '__main__':
    unittest.main()
