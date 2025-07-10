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

_load('asi.hpc_scheduler', 'src/hpc_scheduler.py')
forecast_mod = _load('asi.hpc_forecast_scheduler', 'src/hpc_forecast_scheduler.py')
rl_mod = _load('asi.rl_multi_cluster_scheduler', 'src/rl_multi_cluster_scheduler.py')
HPCForecastScheduler = forecast_mod.HPCForecastScheduler
RLMultiClusterScheduler = rl_mod.RLMultiClusterScheduler
TelemetryLogger = _load('asi.telemetry', 'src/telemetry.py').TelemetryLogger


class TestRLMultiClusterScheduler(unittest.TestCase):
    def test_policy_prefers_cheaper_cluster(self):
        cheap = HPCForecastScheduler()
        exp = HPCForecastScheduler()
        tel_c = TelemetryLogger(carbon_data={'default': 0.2})
        tel_e = TelemetryLogger(carbon_data={'default': 0.5})
        sched = RLMultiClusterScheduler(
            {'cheap': cheap, 'expensive': exp},
            epsilon=0.0,
            telemetry={'cheap': tel_c, 'expensive': tel_e},
        )
        history = [
            {'cluster': 'cheap', 'hour': 0, 'queue_time': 0.1, 'duration': 1.0, 'carbon': 0.2},
            {'cluster': 'expensive', 'hour': 0, 'queue_time': 0.1, 'duration': 1.0, 'carbon': 2.0},
        ]
        for _ in range(5):
            for e in history:
                sched.update_policy(e)
        with patch('asi.rl_multi_cluster_scheduler.submit_job', return_value='jid') as sj, \
             patch('random.random', return_value=1.0), \
             patch('time.time', return_value=0.0):
            cluster, jid = sched.submit_best_rl(['run.sh'])
            self.assertEqual(cluster, 'cheap')
            self.assertEqual(jid, 'jid')
            sj.assert_called_with(['run.sh'], backend='slurm', telemetry=tel_c)
            self.assertAlmostEqual(tel_c.metrics['carbon_saved'], 0.15)
            self.assertEqual(sched.schedule_log[-1][0], 'cheap')


if __name__ == '__main__':
    unittest.main()
