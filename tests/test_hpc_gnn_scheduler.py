import importlib.machinery
import importlib.util
import types
import sys
from unittest.mock import patch
import unittest

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except Exception:  # pragma: no cover - torch optional
    HAS_TORCH = False
    torch = None

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

if HAS_TORCH:
    gnn_mod = _load('asi.hpc_gnn_scheduler', 'src/hpc_gnn_scheduler.py')
    multi_mod = _load('asi.hpc_multi_scheduler', 'src/hpc_multi_scheduler.py')
    GNNForecastScheduler = gnn_mod.GNNForecastScheduler
    MultiClusterScheduler = multi_mod.MultiClusterScheduler


class TestGNNForecastScheduler(unittest.TestCase):
    def test_submit_best_gnn(self):
        if not HAS_TORCH:
            self.skipTest('torch not available')
        a = GNNForecastScheduler(carbon_history=[1.0], cost_history=[1.0])
        b = GNNForecastScheduler(carbon_history=[2.0], cost_history=[2.0])
        sched = MultiClusterScheduler({'a': a, 'b': b})
        with patch.object(gnn_mod.SimpleGNN, 'forward', return_value=torch.tensor([[0.5, 0.5], [1.0, 1.0]])), \
             patch('time.sleep') as sl, \
             patch('asi.hpc_multi_scheduler.submit_job', return_value='jid') as sj:
            cluster, jid = sched.submit_best(['run.sh'], max_delay=3600.0)
            sl.assert_not_called()
            sj.assert_called_with(['run.sh'], backend='slurm')
            self.assertEqual(cluster, 'a')
            self.assertEqual(jid, 'jid')


if __name__ == '__main__':
    unittest.main()
