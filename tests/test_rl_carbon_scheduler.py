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

torch_stub = types.SimpleNamespace(
    cuda=types.SimpleNamespace(
        is_available=lambda: False,
        memory_allocated=lambda: 0,
        get_device_properties=lambda _: types.SimpleNamespace(total_memory=1),
    )
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
_load('asi.adaptive_micro_batcher', 'src/adaptive_micro_batcher.py')
gpu_sched_stub = types.ModuleType('asi.gpu_aware_scheduler')
gpu_sched_stub.GPUAwareScheduler = object
sys.modules['asi.gpu_aware_scheduler'] = gpu_sched_stub
accel_stub = types.ModuleType('asi.schedulers')
accel_stub.AcceleratorScheduler = object
sys.modules['asi.schedulers'] = accel_stub
_load('asi.enclave_runner', 'src/enclave_runner.py')
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
RLCarbonScheduler = _load('asi.rl_schedulers', 'src/rl_schedulers.py').RLCarbonScheduler
hpc_mod = _load('asi.hpc_schedulers', 'src/hpc_schedulers.py')
submit_job = hpc_mod.submit_job
dt_mod = _load('asi.distributed_trainer', 'src/distributed_trainer.py')
DistributedTrainer = dt_mod.DistributedTrainer
MemoryConfig = dt_mod.MemoryConfig


class TestRLCarbonScheduler(unittest.TestCase):
    def test_wait_and_run(self):
        logger = TelemetryLogger(interval=0.05, carbon_data={'default': 0.8})
        hist = [(0.8, 1.0), (0.2, 1.0)]
        sched = RLCarbonScheduler(hist, telemetry=logger, check_interval=0.05)
        job_id = []

        def run():
            job_id.append(sched.submit_job(['job.sh'], backend='slurm'))

        with patch('asi.rl_schedulers.submit_job', return_value='ok') as sj:
            t = threading.Thread(target=run)
            t.start()
            time.sleep(0.1)
            logger.carbon_data['default'] = 0.2
            t.join(timeout=0.3)
            self.assertEqual(job_id[0], 'ok')
            self.assertGreaterEqual(sj.call_count, 1)
            self.assertGreater(sched.telemetry.metrics['wait_time'], 0.0)

    def test_trainer_integration(self):
        logger = TelemetryLogger(interval=0.05)
        sched = RLCarbonScheduler([], telemetry=logger)

        def dummy(mem, step, comp=None):
            pass

        cfg = MemoryConfig(dim=2, compressed_dim=1, capacity=2)
        with patch.object(sched, 'submit_job', return_value='jid') as sj:
            trainer = DistributedTrainer(dummy, cfg, '/tmp', hpc_backend='slurm', scheduler=sched)
            trainer.run(steps=1)
            sj.assert_called()


if __name__ == '__main__':
    unittest.main()

