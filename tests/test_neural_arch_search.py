import unittest
import importlib.machinery
import importlib.util
import types
import sys
import random

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg
sys.modules['psutil'] = types.SimpleNamespace(
    cpu_percent=lambda interval=None: 0.0,
    virtual_memory=lambda: types.SimpleNamespace(percent=0.0),
    net_io_counters=lambda: types.SimpleNamespace(bytes_sent=0, bytes_recv=0),
)
sys.modules['torch'] = types.SimpleNamespace(
    cuda=types.SimpleNamespace(
        utilization=lambda: 0.0,
        is_available=lambda: False,
        memory_allocated=lambda: 0,
        get_device_properties=lambda _: types.SimpleNamespace(total_memory=1),
    )
)
sys.modules['asi.carbon_tracker'] = types.SimpleNamespace(
    CarbonFootprintTracker=type(
        'CF',
        (),
        {
            '__init__': lambda self, **kw: None,
            'start': lambda self: None,
            'stop': lambda self: None,
            'get_stats': lambda self: {},
        },
    )
)
sys.modules['asi.memory_event_detector'] = types.SimpleNamespace(
    MemoryEventDetector=type(
        'MED',
        (),
        {
            '__init__': lambda self, **kw: None,
            'update': lambda self, x: [],
            'events': [],
        },
    )
)


def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = name.rpartition('.')[0]
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod

TelemetryLogger = _load('asi.telemetry', 'src/telemetry.py').TelemetryLogger
DistributedArchSearch = _load('asi.neural_arch_search', 'src/neural_arch_search.py').DistributedArchSearch


class TestNeuralArchSearch(unittest.TestCase):
    def test_search_selects_best(self):
        space = {"layers": [1, 2], "hidden": [8, 16]}
        random.seed(0)

        def score(cfg):
            return cfg["layers"] * cfg["hidden"]

        search = DistributedArchSearch(space, score, max_workers=1)
        best, val = search.search(num_samples=4)
        self.assertEqual(best["layers"], 2)
        self.assertEqual(best["hidden"], 16)
        self.assertEqual(val, 32)

    def test_energy_weight_affects_ranking(self):
        space = {"layers": [1, 2]}
        random.seed(0)
        energies = [0.0, 0.0, 3.0, 3.0, 3.1]

        class DummyLogger(TelemetryLogger):
            def __init__(self, vals):
                super().__init__(interval=0.01)
                self.vals = vals
                self.idx = 0

            def start(self):
                pass

            def stop(self):
                pass

            def get_stats(self):
                val = self.vals[self.idx]
                if self.idx < len(self.vals) - 1:
                    self.idx += 1
                return {"energy_kwh": val}

        def score(cfg):
            return float(cfg["layers"]) * 10

        tel = DummyLogger(energies)
        search = DistributedArchSearch(space, score, max_workers=1, telemetry=tel, energy_weight=5.0)
        best, _ = search.search(num_samples=2)
        self.assertEqual(best["layers"], 1)


if __name__ == "__main__":
    unittest.main()
