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
NeuroevolutionSearch = _load('asi.neuroevolution_search', 'src/neuroevolution_search.py').NeuroevolutionSearch


class TestNeuroevolutionSearch(unittest.TestCase):
    def test_evolution_improves_loss(self):
        space = {"layers": [1, 2, 3, 4], "hidden": [4, 8, 12, 16]}
        random.seed(2)

        def score(cfg):
            loss = (cfg["layers"] - 4) ** 2 + (cfg["hidden"] - 12) ** 2
            return -float(loss)

        search = NeuroevolutionSearch(
            space,
            score,
            population_size=4,
            mutation_rate=0.5,
            crossover_rate=0.5,
            telemetry=TelemetryLogger(interval=0.01),
        )
        best, best_score = search.search(generations=5)
        first_loss = -search.history[0]
        final_loss = -search.history[-1]
        self.assertLess(final_loss, first_loss)
        self.assertEqual(best["layers"], 4)
        self.assertEqual(best["hidden"], 12)
        self.assertGreater(best_score, -0.1)


if __name__ == "__main__":
    unittest.main()
