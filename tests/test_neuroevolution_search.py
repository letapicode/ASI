import unittest
import importlib.machinery
import importlib.util
import types
import sys
import random

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

NeuroevolutionSearch = _load('asi.neuroevolution_search', 'src/neuroevolution_search.py').NeuroevolutionSearch


class TestNeuroevolutionSearch(unittest.TestCase):
    def test_evolve_finds_best(self):
        space = {"x": [0, 1], "y": [1, 2]}
        random.seed(0)

        def score(cfg):
            return cfg["x"] + cfg["y"]

        search = NeuroevolutionSearch(space, score, population_size=4, mutation_rate=0.5)
        best, val = search.evolve(generations=2)
        self.assertEqual(best["x"], 1)
        self.assertEqual(best["y"], 2)
        self.assertEqual(val, 3)


if __name__ == "__main__":
    unittest.main()
