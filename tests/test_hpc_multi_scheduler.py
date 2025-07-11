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

hpc_mod = _load('asi.hpc_schedulers', 'src/hpc_schedulers.py')
forecast_mod = _load('asi.hpc_forecast_scheduler', 'src/hpc_forecast_scheduler.py')
mod = _load('asi.hpc_multi_scheduler', 'src/hpc_multi_scheduler.py')
HPCForecastScheduler = forecast_mod.HPCForecastScheduler
MultiClusterScheduler = mod.MultiClusterScheduler
TelemetryLogger = mod.TelemetryLogger


class TestMultiClusterScheduler(unittest.TestCase):
    def test_submit_best(self):
        a = HPCForecastScheduler()
        b = HPCForecastScheduler(backend='k8s')
        tel_a = TelemetryLogger(carbon_data={'default': 0.5})
        tel_b = TelemetryLogger(carbon_data={'default': 0.2})
        sched = MultiClusterScheduler(
            {'a': a, 'b': b}, telemetry={'a': tel_a, 'b': tel_b}
        )
        with patch('asi.hpc_forecast_scheduler.arima_forecast', side_effect=[[10, 1], [1.0, 0.2], [5, 0.5], [0.5, 0.1]]), \
             patch('time.sleep') as sl, \
             patch('subprocess.run') as sp:
            sp.return_value = types.SimpleNamespace(stdout='jid', returncode=0)
            cluster, jid = sched.submit_best(['run.sh'], max_delay=7200.0)
            sp.assert_called()
            self.assertIn(cluster, {'a', 'b'})
            self.assertEqual(jid, 'jid')
            baseline = (tel_a.get_live_carbon_intensity() + tel_b.get_live_carbon_intensity()) / 2
            chosen_tel = tel_a if cluster == 'a' else tel_b
            expected = (baseline - chosen_tel.get_live_carbon_intensity())
            self.assertAlmostEqual(chosen_tel.metrics['carbon_saved'], expected)
            self.assertEqual(sched.schedule_log[-1][0], cluster)


if __name__ == '__main__':
    unittest.main()
