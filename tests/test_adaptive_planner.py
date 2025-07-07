import unittest
import importlib.machinery
import importlib.util

loader = importlib.machinery.SourceFileLoader('adaptive_planner', 'src/adaptive_planner.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
adaptive_planner = importlib.util.module_from_spec(spec)
loader.exec_module(adaptive_planner)
AdaptivePlanner = adaptive_planner.AdaptivePlanner
MetaOptimizer = adaptive_planner.MetaOptimizer


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

    def test_meta_optimizer_updates_model(self):
        import torch
        from torch import nn

        torch.manual_seed(0)
        model = nn.Linear(1, 1, bias=False)

        def train_step(m: nn.Module, data):
            x, y = data
            pred = m(x)
            return ((pred - y) ** 2).mean()

        def make_task(a: float):
            x = torch.randn(4, 1)
            y = a * x
            return (x, y)

        opt = MetaOptimizer(train_step, adapt_lr=0.05, meta_lr=0.05)
        planner = AdaptivePlanner(
            quality_score,
            epsilon=0.0,
            meta_optimizer=opt,
            model=model,
        )
        before = model.weight.clone()
        tasks = [make_task(1.0), make_task(2.0)]
        planner.meta_optimizer.meta_step(model, tasks)
        after = model.weight
        self.assertFalse(torch.equal(before, after))


if __name__ == '__main__':
    unittest.main()
