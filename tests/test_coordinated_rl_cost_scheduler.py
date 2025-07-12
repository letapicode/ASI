import importlib.machinery
import importlib.util
import types
import sys
import threading
import time
from unittest.mock import patch
import unittest

psutil_stub = types.SimpleNamespace(
    cpu_percent=lambda interval=None: 50.0,
    virtual_memory=lambda: types.SimpleNamespace(percent=10.0),
    net_io_counters=lambda: types.SimpleNamespace(bytes_sent=0, bytes_recv=0),
)
sys.modules['psutil'] = psutil_stub
sys.modules['numpy'] = types.SimpleNamespace(asarray=lambda x, dtype=None: x)
sys.modules['statsmodels'] = types.ModuleType('statsmodels')
sys.modules['statsmodels.tsa'] = types.ModuleType('statsmodels.tsa')
sys.modules['statsmodels.tsa.arima'] = types.ModuleType('statsmodels.tsa.arima')
sm_arima = types.ModuleType('statsmodels.tsa.arima.model')
sm_arima.ARIMA = object
sys.modules['statsmodels.tsa.arima.model'] = sm_arima
requests_stub = types.ModuleType('requests')
requests_stub.get = lambda *a, **k: None
sys.modules['requests'] = requests_stub
dm_stub = types.ModuleType('asi.distributed_memory')
class _DM:
    def __init__(self, *a, **kw):
        pass
    def save(self, *a, **kw):
        pass
    def add(self, *a, **kw):
        pass

dm_stub.DistributedMemory = _DM
sys.modules['asi.distributed_memory'] = dm_stub

torch_stub = types.SimpleNamespace(
    cuda=types.SimpleNamespace(
        is_available=lambda: False,
        memory_allocated=lambda: 0,
        get_device_properties=lambda _: types.SimpleNamespace(total_memory=1),
    ),
    nn=types.SimpleNamespace(Module=object)
)
sys.modules['torch'] = torch_stub

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

_load('asi.carbon_tracker', 'src/carbon_tracker.py')
_load('asi.memory_event_detector', 'src/memory_event_detector.py')
TelemetryLogger = _load('asi.telemetry', 'src/telemetry.py').TelemetryLogger
_load('asi.hpc_schedulers', 'src/hpc_schedulers.py')
rl_mod = _load('asi.rl_schedulers', 'src/rl_schedulers.py')
CoordinatedRLCostScheduler = rl_mod.CoordinatedRLCostScheduler
base_mod = _load('asi.hpc_schedulers', 'src/hpc_schedulers.py')
make_scheduler = base_mod.make_scheduler


class TestCoordinatedRLCostScheduler(unittest.TestCase):
    def test_peer_exchange(self):
        sched1 = CoordinatedRLCostScheduler({'c': make_scheduler('arima')}, epsilon=0.0)
        sched2 = CoordinatedRLCostScheduler({'c': make_scheduler('arima')}, epsilon=0.0)
        sched1.q1[(0, 0, 0)] = 1.0
        sched2.q1[(0, 0, 0)] = 3.0
        sched1.register_peer('p', sched2)
        with patch.object(sched1, '_bucket', return_value=0), \
             patch.object(sched1, '_train', return_value=None), \
             patch('asi.forecast_strategies.arima_forecast', return_value=[0.0]), \
             patch('asi.rl_schedulers.submit_job', return_value='jid') as sj:
            cluster, jid = sched1.submit_best(['run'], max_delay=0.0)
            self.assertEqual(cluster, 'c')
            self.assertEqual(jid, 'jid')
            sj.assert_called()
        # q-value should average with peer
        self.assertAlmostEqual(sched1.q1[(0, 0, 0)], 2.0)

    def test_wait_loop_with_peers(self):
        logger = TelemetryLogger(interval=0.05,
                                 carbon_data={'default': 1.0},
                                 energy_price_data={'default': 1.0})
        hist = make_scheduler('arima', carbon_history=[1.0, 0.2], cost_history=[1.0, 0.1])
        sched1 = CoordinatedRLCostScheduler({'c': hist}, check_interval=0.05)
        sched2 = CoordinatedRLCostScheduler({'c': hist}, check_interval=0.05)
        sched1.register_peer('p', sched2)
        job = []

        def run():
            job.append(sched1.submit_best(['job.sh'], max_delay=0.0))

        with patch('asi.rl_schedulers.submit_job', return_value='ok') as sj:
            t = threading.Thread(target=run)
            t.start()
            time.sleep(0.1)
            logger.carbon_data['default'] = 0.2
            logger.energy_price_data['default'] = 0.1
            t.join(timeout=0.3)
            self.assertTrue(job)
            sj.assert_called()


if __name__ == '__main__':
    unittest.main()
