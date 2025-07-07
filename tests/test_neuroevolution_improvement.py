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


class TestNeuroevolutionImprovement(unittest.TestCase):
    def test_loss_improves_over_generations(self):
        random.seed(2)

        # target hidden size outside initial population to force improvement
        target = 5

        def score(cfg):
            hidden = cfg['hidden']
            loss = (hidden - target) ** 2
            return -float(loss)

        space = {'hidden': [1, 2, 3, 4, 5]}
        search = NeuroevolutionSearch(space, score, population_size=4, mutation_rate=1.0)
        _, _, history = search.evolve(generations=3, return_history=True)
        losses = [-s for s in history]
        self.assertLess(losses[-1], losses[0])


if __name__ == '__main__':
    unittest.main()
