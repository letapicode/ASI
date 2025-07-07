import importlib.util
import types
import sys
import time
import threading
import unittest
from unittest.mock import patch

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
gc_stub = types.ModuleType('asi.gradient_compression')
class _GCfg:
    def __init__(self, topk=None, bits=None):
        self.topk = topk
        self.bits = bits

class _GComp:
    def __init__(self, cfg):
        self.cfg = cfg
    def compress(self, g):
        return g

gc_stub.GradientCompressionConfig = _GCfg
gc_stub.GradientCompressor = _GComp
sys.modules['asi.gradient_compression'] = gc_stub
requests_stub = types.ModuleType('requests')
requests_stub.get = lambda *a, **k: None
sys.modules['requests'] = requests_stub
dm_stub = types.ModuleType('asi.distributed_memory')
class _DM:
    def __init__(self, *a, **kw):
        pass
    def save(self, *a, **k):
        pass
    def add(self, *a, **k):
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
_load('asi.hpc_scheduler', 'src/hpc_scheduler.py')
rl_mod = _load('asi.rl_cost_scheduler', 'src/rl_cost_scheduler.py')
RLCostScheduler = rl_mod.RLCostScheduler
hfc_mod = _load('asi.hpc_forecast_scheduler', 'src/hpc_forecast_scheduler.py')
HPCForecastScheduler = hfc_mod.HPCForecastScheduler

dt_mod = _load('asi.distributed_trainer', 'src/distributed_trainer.py')
DistributedTrainer = dt_mod.DistributedTrainer
MemoryConfig = dt_mod.MemoryConfig


class TestRLCostScheduler(unittest.TestCase):
    def test_wait_and_run(self):
        logger = TelemetryLogger(interval=0.05,
                                 carbon_data={'default': 1.0},
                                 energy_price_data={'default': 1.0})
        hist = HPCForecastScheduler(carbon_history=[1.0, 0.2], cost_history=[1.0, 0.1])
        sched = RLCostScheduler({'c': hist}, check_interval=0.05)
        job = []

        def run():
            job.append(sched.submit_best(['job.sh'], max_delay=0.0))

        with patch('asi.rl_cost_scheduler.submit_job', return_value='ok') as sj:
            t = threading.Thread(target=run)
            t.start()
            time.sleep(0.1)
            logger.carbon_data['default'] = 0.2
            logger.energy_price_data['default'] = 0.1
            t.join(timeout=0.3)
            self.assertTrue(job)
            sj.assert_called()

    def test_trainer_integration(self):
        logger = TelemetryLogger(interval=0.05)
        sched = RLCostScheduler({'a': HPCForecastScheduler()})

        def dummy(mem, step, comp=None):
            pass

        cfg = MemoryConfig(dim=2, compressed_dim=1, capacity=2)
        with patch.object(sched, 'submit_best', return_value=('a', 'jid')) as sb:
            trainer = DistributedTrainer(dummy, cfg, '/tmp', hpc_backend='slurm', scheduler=sched)
            trainer.run(steps=1)
            sb.assert_called()


if __name__ == '__main__':
    unittest.main()
