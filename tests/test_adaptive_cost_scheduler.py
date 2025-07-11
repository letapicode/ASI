import importlib.machinery
import importlib.util
import types
import sys
import os
import tempfile
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
sys.modules['numpy'] = types.SimpleNamespace(asarray=lambda x, dtype=None: x)
sys.modules['statsmodels'] = types.ModuleType('statsmodels')
sys.modules['statsmodels.tsa'] = types.ModuleType('statsmodels.tsa')
sys.modules['statsmodels.tsa.arima'] = types.ModuleType('statsmodels.tsa.arima')
sm_arima = types.ModuleType('statsmodels.tsa.arima.model')
sm_arima.ARIMA = object
sys.modules['statsmodels.tsa.arima.model'] = sm_arima

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

base = _load('asi.hpc_base_scheduler', 'src/hpc_base_scheduler.py')
strat = _load('asi.forecast_strategies', 'src/forecast_strategies.py')
multi = _load('asi.hpc_multi_scheduler', 'src/hpc_multi_scheduler.py')
mod = _load('asi.adaptive_cost_scheduler', 'src/adaptive_cost_scheduler.py')
AdaptiveCostScheduler = mod.AdaptiveCostScheduler
make_scheduler = base.make_scheduler


class TestAdaptiveCostScheduler(unittest.TestCase):
    def test_submit_best(self):
        a = make_scheduler('arima')
        b = make_scheduler('arima', backend='k8s')
        sched = AdaptiveCostScheduler({'a': a, 'b': b})
        with patch('asi.forecast_strategies.arima_forecast', side_effect=[[10, 1], [1.0, 0.2], [5, 0.5], [0.5, 0.1]]), \
             patch.object(sched, '_policy', return_value=0) as pol, \
             patch('time.sleep') as sl, \
             patch('asi.adaptive_cost_scheduler.submit_job', return_value='jid') as sj:
            cluster, jid = sched.submit_best(['run.sh'], max_delay=7200.0)
            sl.assert_called_with(3600.0)
            sj.assert_called_with(['run.sh'], backend='k8s')
            self.assertEqual(cluster, 'b')
            self.assertEqual(jid, 'jid')
            pol.assert_called()

    def test_persist_qtable(self):
        a = make_scheduler('arima', carbon_history=[1, 2], cost_history=[2, 1])
        b = make_scheduler('arima', backend='k8s', carbon_history=[3, 4], cost_history=[4, 3])
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'q.json')
            sched = AdaptiveCostScheduler({'a': a, 'b': b}, qtable_path=path)
            self.assertTrue(os.path.exists(path))
            q_pre = dict(sched.q)
            sched2 = AdaptiveCostScheduler({'a': a, 'b': b}, qtable_path=path)
            self.assertEqual(q_pre, sched2.q)


if __name__ == '__main__':
    unittest.main()
