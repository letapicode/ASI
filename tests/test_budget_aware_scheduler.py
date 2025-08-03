import unittest
import importlib.machinery
import importlib.util
import types
import sys

torch_stub = types.SimpleNamespace(
    cuda=types.SimpleNamespace(
        utilization=lambda: 0.0,
        is_available=lambda: False,
        memory_allocated=lambda: 0,
    )
)
sys.modules['torch'] = torch_stub

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg

loader_tel = importlib.machinery.SourceFileLoader('asi.telemetry', 'src/telemetry.py')
spec_tel = importlib.util.spec_from_loader(loader_tel.name, loader_tel)
mod_tel = importlib.util.module_from_spec(spec_tel)
mod_tel.__package__ = 'asi'
sys.modules[loader_tel.name] = mod_tel
loader_tel.exec_module(mod_tel)

loader_cb = importlib.machinery.SourceFileLoader('asi.compute_budget_tracker', 'src/compute_budget_tracker.py')
spec_cb = importlib.util.spec_from_loader(loader_cb.name, loader_cb)
mod_cb = importlib.util.module_from_spec(spec_cb)
mod_cb.__package__ = 'asi'
sys.modules[loader_cb.name] = mod_cb
loader_cb.exec_module(mod_cb)
ComputeBudgetTracker = mod_cb.ComputeBudgetTracker

loader_sched = importlib.machinery.SourceFileLoader('asi.schedulers', 'src/schedulers.py')
spec_sched = importlib.util.spec_from_loader(loader_sched.name, loader_sched)
mod_sched = importlib.util.module_from_spec(spec_sched)
mod_sched.__package__ = 'asi'
sys.modules[loader_sched.name] = mod_sched
loader_sched.exec_module(mod_sched)
BudgetAwareScheduler = mod_sched.BudgetAwareScheduler


class DummyCfg:
    def __init__(self, lr: float, batch_size: int) -> None:
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = 1


class TestBudgetAwareScheduler(unittest.TestCase):
    def test_schedule(self):
        cfg = DummyCfg(lr=0.1, batch_size=4)
        tracker = ComputeBudgetTracker(0.0)
        sched = BudgetAwareScheduler(tracker)
        sched.schedule_step(cfg)
        self.assertEqual(cfg.batch_size, 2)
        self.assertAlmostEqual(cfg.lr, 0.05)


if __name__ == '__main__':
    unittest.main()
