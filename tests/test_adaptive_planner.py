import unittest
import importlib.machinery
import importlib.util

loader = importlib.machinery.SourceFileLoader('adaptive_planner', 'src/adaptive_planner.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
adaptive_planner = importlib.util.module_from_spec(spec)
loader.exec_module(adaptive_planner)
AdaptivePlanner = adaptive_planner.AdaptivePlanner


def quality_score(strategy: str) -> float:
    strategy = strategy.lower()
    if 'refactor' in strategy:
        return 2.0
    if 'replace' in strategy:
        return 1.0
    return 0.0


class TestAdaptivePlanner(unittest.TestCase):
    def test_ranking(self):
        planner = AdaptivePlanner(quality_score, epsilon=0.0)
        strategies = ['rollback', 'replace var', 'refactor loops']
        ranked = planner.rank_strategies(strategies)
        self.assertEqual(ranked[0][0], 'refactor loops')
        self.assertGreater(ranked[0][1], ranked[-1][1])

    def test_learning_improves_quality(self):
        import random

        random.seed(1)
        planner = AdaptivePlanner(quality_score, epsilon=1.0)
        strategies = ['replace foo', 'rollback', 'refactor bar']

        action0 = planner.agent.select_action(0)
        first = next(s for s in strategies if s.startswith(action0))

        planner.rank_strategies(strategies)
        planner.agent.epsilon = 0.0
        second = planner.best_strategy(strategies)
        self.assertLess(quality_score(first), quality_score(second))


if __name__ == '__main__':
    unittest.main()
