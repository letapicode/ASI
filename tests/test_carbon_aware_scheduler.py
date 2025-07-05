import importlib.machinery
import importlib.util
import types
import sys
import time
from unittest.mock import patch
import unittest

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg


def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = name.rpartition('.')[0]
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod

TelemetryLogger = _load('asi.telemetry', 'src/telemetry.py').TelemetryLogger
CarbonAwareScheduler = _load('asi.carbon_aware_scheduler', 'src/carbon_aware_scheduler.py').CarbonAwareScheduler
hpc_mod = _load('asi.hpc_scheduler', 'src/hpc_scheduler.py')
submit_job = hpc_mod.submit_job


class TestCarbonAwareScheduler(unittest.TestCase):
    def test_immediate_run(self):
        logger = TelemetryLogger(interval=0.05, carbon_data={'default': 0.3})
        sched = CarbonAwareScheduler(0.5, telemetry=logger, check_interval=0.05)
        with patch.object(hpc_mod, 'submit_job', return_value='id') as run:
            jid = sched.submit_job(['job.sh'])
            self.assertEqual(jid, 'id')
            run.assert_called_once()
        sched.stop()

    def test_queue_and_release(self):
        logger = TelemetryLogger(interval=0.05, carbon_data={'default': 0.7})
        sched = CarbonAwareScheduler(0.5, telemetry=logger, check_interval=0.05)
        with patch.object(hpc_mod, 'submit_job', return_value='ok') as run:
            res = sched.submit_job(['job.sh'])
            self.assertEqual(res, 'QUEUED')
            self.assertEqual(run.call_count, 0)
            logger.carbon_data['default'] = 0.3
            time.sleep(0.1)
            sched.stop()
            self.assertGreaterEqual(run.call_count, 1)


if __name__ == '__main__':
    unittest.main()
