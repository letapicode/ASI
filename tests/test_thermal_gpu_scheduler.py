import importlib.machinery
import importlib.util
import types
import sys
import time
from unittest.mock import patch
import unittest

psutil_stub = types.SimpleNamespace(
    cpu_percent=lambda interval=None: 0.0,
    virtual_memory=lambda: types.SimpleNamespace(percent=0.0),
    net_io_counters=lambda: types.SimpleNamespace(bytes_sent=0, bytes_recv=0),
)
sys.modules['psutil'] = psutil_stub

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg



def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = name.rpartition('.')[0]
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod

_load('asi.carbon_tracker', 'src/carbon_tracker.py')
_load('asi.memory_event_detector', 'src/memory_event_detector.py')
TelemetryLogger = _load('asi.telemetry', 'src/telemetry.py').TelemetryLogger
accel_mod = _load('asi.accelerator_scheduler', 'src/accelerator_scheduler.py')
ThermalScheduler = _load('asi.thermal_gpu_scheduler', 'src/thermal_gpu_scheduler.py').ThermalGPUAwareScheduler
hpc_mod = _load('asi.hpc_scheduler', 'src/hpc_scheduler.py')


class TestThermalScheduler(unittest.TestCase):
    def test_submit_job_defer(self):
        logger = TelemetryLogger(interval=0.05)
        logger.gpu_temperature = lambda index=0: 90.0
        with patch('subprocess.run') as run:
            run.return_value = types.SimpleNamespace(returncode=0, stdout='id', stderr='')
            jid = hpc_mod.submit_job(['job.sh'], telemetry=logger, max_temp=80.0)
            self.assertEqual(jid, 'DEFERRED')
            run.assert_not_called()

    def test_throttle(self):
        accel_mod.psutil.cpu_percent = lambda interval=None: 0.0
        if hasattr(accel_mod, 'torch') and accel_mod.torch is not None:
            accel_mod.torch.cuda.is_available = lambda: True
            accel_mod.torch.cuda.memory_allocated = lambda: 0
            class Props:
                total_memory = 1
            accel_mod.torch.cuda.get_device_properties = lambda idx: Props()
        temps = [90.0]
        logger = TelemetryLogger(interval=0.05)
        logger.gpu_temperature = lambda index=0: temps[0]
        sched = ThermalScheduler(max_temp=80.0, check_interval=0.05, telemetry=logger)
        ran = []
        sched.add(lambda: ran.append(1))
        time.sleep(0.1)
        self.assertEqual(len(ran), 0)
        temps[0] = 60.0
        time.sleep(0.1)
        self.assertGreaterEqual(len(ran), 1)


if __name__ == '__main__':
    unittest.main()
