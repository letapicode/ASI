import unittest
import importlib.machinery
import importlib.util
import types
import sys
import torch

src_pkg = types.ModuleType('src')
src_pkg.__path__ = ['src']
src_pkg.__spec__ = importlib.machinery.ModuleSpec('src', None, is_package=True)
sys.modules['src'] = src_pkg
loader = importlib.machinery.SourceFileLoader('src.edge_rl_trainer', 'src/edge_rl_trainer.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
mod = importlib.util.module_from_spec(spec)
mod.__package__ = 'src'
sys.modules['src.edge_rl_trainer'] = mod
loader.exec_module(mod)
EdgeRLTrainer = mod.EdgeRLTrainer
EdgeRLTrainer = mod.EdgeRLTrainer

cb_loader = importlib.machinery.SourceFileLoader('src.compute_budget_tracker', 'src/compute_budget_tracker.py')
cb_spec = importlib.util.spec_from_loader(cb_loader.name, cb_loader)
cbm = importlib.util.module_from_spec(cb_spec)
cbm.__package__ = 'src'
sys.modules['src.compute_budget_tracker'] = cbm
cb_loader.exec_module(cbm)
ComputeBudgetTracker = cbm.ComputeBudgetTracker
ComputeBudgetTracker = cbm.ComputeBudgetTracker
tel_loader = importlib.machinery.SourceFileLoader('src.telemetry', 'src/telemetry.py')
tel_spec = importlib.util.spec_from_loader(tel_loader.name, tel_loader)
tel_mod = importlib.util.module_from_spec(tel_spec)
tel_mod.__package__ = 'src'
sys.modules['src.telemetry'] = tel_mod
tel_loader.exec_module(tel_mod)
TelemetryLogger = tel_mod.TelemetryLogger

class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(2,2)
    def forward(self,x):
        return self.l(x)

class TestEdgeRLTrainer(unittest.TestCase):
    def test_budget(self):
        model = Toy()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        budget = ComputeBudgetTracker(0.0)
        trainer = EdgeRLTrainer(model, opt, budget)
        data = [(torch.randn(1,2), torch.randn(1,2)) for _ in range(3)]
        steps = trainer.train(data)
        self.assertEqual(steps, 0)

    def test_power_logging(self):
        class DummyLogger(TelemetryLogger):
            def __init__(self):
                super().__init__(interval=0.01)
                self.vals = [0.0, 0.1]

            def start(self):
                pass

            def stop(self):
                pass

            def get_stats(self):
                v = self.vals.pop(0) if self.vals else 0.1
                return {"energy_kwh": v}

        logger = DummyLogger()
        budget = ComputeBudgetTracker(float("inf"), telemetry=logger)
        model = Toy()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        trainer = EdgeRLTrainer(model, opt, budget, use_loihi=True)
        data = [(torch.zeros(1, 2), torch.zeros(1, 2))]
        trainer.train(data)
        self.assertGreater(trainer.power_usage["loihi"], 0)

if __name__ == '__main__':
    unittest.main()
