import importlib.machinery
import importlib.util
import types
import sys
import unittest
from unittest.mock import patch


pkg = types.ModuleType('asi')
pkg.__path__ = ['src']
pkg.__spec__ = importlib.machinery.ModuleSpec('asi', None, is_package=True)
sys.modules['asi'] = pkg

requests_stub = types.ModuleType('requests')
requests_stub.get = lambda *a, **kw: None
sys.modules['requests'] = requests_stub
pil_stub = types.ModuleType('PIL')
pil_stub.Image = types.SimpleNamespace(open=lambda *a, **kw: None)
sys.modules['PIL'] = pil_stub
sys.modules['aiohttp'] = types.ModuleType('aiohttp')


def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = name.rpartition('.')[0]
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod


_load('asi.telemetry', 'src/telemetry.py')
_load('asi.hpc_schedulers', 'src/hpc_schedulers.py')
carbon_mod = _load('asi.carbon_hpc_scheduler', 'src/carbon_hpc_scheduler.py')
CarbonAwareScheduler = carbon_mod.CarbonAwareScheduler

_load('asi.carbon_tracker', 'src/carbon_tracker.py')

di = _load('asi.data_ingest', 'src/data_ingest.py')

cadi = _load('asi.carbon_aware_dataset_ingest', 'src/carbon_aware_dataset_ingest.py')
CarbonAwareDatasetIngest = cadi.CarbonAwareDatasetIngest


class TestCarbonAwareDatasetIngest(unittest.TestCase):
    def test_waits_until_green(self):
        sched = CarbonAwareScheduler(threshold=300.0, check_interval=0.01)
        ingest = CarbonAwareDatasetIngest(sched)

        calls = []

        def fake_intensity(region=None):
            calls.append(1)
            return 500.0 if len(calls) == 1 else 200.0

        with patch('asi.carbon_hpc_scheduler.get_carbon_intensity', side_effect=fake_intensity), \
             patch('asi.carbon_aware_dataset_ingest.get_carbon_intensity', side_effect=fake_intensity), \
             patch('asi.carbon_aware_dataset_ingest.download_triples', return_value=['ok']) as dl, \
             patch('asi.carbon_aware_dataset_ingest.time.sleep', lambda s: None):
            res = ingest.download_when_green(['t'], ['i'], ['a'], '/tmp')
        self.assertEqual(res, ['ok'])
        self.assertEqual(dl.call_count, 1)
        self.assertGreaterEqual(len(calls), 2)

    def test_optimal_time(self):
        sched = CarbonAwareScheduler(threshold=300.0, check_interval=0.01)
        ingest = CarbonAwareDatasetIngest(sched)

        with patch('asi.carbon_hpc_scheduler.get_hourly_forecast', return_value=[400, 200]), \
             patch('asi.carbon_aware_dataset_ingest.get_hourly_forecast', return_value=[400, 200]), \
             patch('asi.carbon_aware_dataset_ingest.time.sleep', lambda s: None), \
             patch('asi.carbon_aware_dataset_ingest.download_triples', return_value=['x']) as dl:
            res = ingest.download_at_optimal_time(['t'], ['i'], ['a'], '/tmp')
        self.assertEqual(res, ['x'])
        self.assertEqual(dl.call_count, 1)


