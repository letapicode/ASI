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
psutil_stub.sensors_battery = lambda: None
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

requests_stub = types.ModuleType('requests')
requests_stub.get = lambda *a, **kw: types.SimpleNamespace(json=lambda: {}, raise_for_status=lambda: None)
sys.modules['requests'] = requests_stub

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
ca_mod = _load('asi.carbon_aware_scheduler', 'src/carbon_aware_scheduler.py')
rl_mod = _load('asi.rl_schedulers', 'src/rl_schedulers.py')
base_mod = _load('asi.hpc_schedulers', 'src/hpc_schedulers.py')
strat_mod = _load('asi.forecast_strategies', 'src/forecast_strategies.py')
meta_mod = _load('asi.meta_scheduler', 'src/meta_scheduler.py')
hpc_mod = _load('asi.hpc_schedulers', 'src/hpc_schedulers.py')

TelemetryLogger = telemetry_mod.TelemetryLogger
CarbonAwareScheduler = ca_mod.CarbonAwareScheduler
RLCarbonScheduler = rl_mod.RLCarbonScheduler
make_scheduler = base_mod.make_scheduler
MetaScheduler = meta_mod.MetaScheduler


class TestMetaScheduler(unittest.TestCase):
    def test_choose_best_scheduler(self) -> None:
        ca = CarbonAwareScheduler(1.0, telemetry=TelemetryLogger(interval=0.05), check_interval=0.05)
        rl = RLCarbonScheduler([], telemetry=TelemetryLogger(interval=0.05), check_interval=0.05)
        fc = make_scheduler('arima')
        tf = make_scheduler('arima')
        sched = MetaScheduler({'ca': ca, 'rl': rl, 'fc': fc, 'tf': tf})
        sched.record_result('rl', True, 0.5, 0.5)
        sched.record_result('ca', True, 1.0, 1.0)
        sched.record_result('fc', True, 2.0, 2.0)
        sched.record_result('tf', True, 2.0, 2.0)
        with patch.object(hpc_mod, 'submit_job', return_value='jid'), \
             patch.object(rl_mod, 'submit_job', return_value='jid'), \
             patch('asi.hpc_schedulers.submit_job', return_value='jid'):
            name, jid = sched.submit_best(['run.sh'])
            self.assertEqual(name, 'rl')
            self.assertEqual(jid, 'jid')
        ca.stop()


if __name__ == '__main__':  # pragma: no cover
    unittest.main()

