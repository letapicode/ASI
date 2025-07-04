import importlib.machinery
import importlib.util
import asyncio
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

loader = importlib.machinery.SourceFileLoader(
    'multi_agent_coordinator', 'src/multi_agent_coordinator.py'
)
spec = importlib.util.spec_from_loader(loader.name, loader)
coordinator_mod = importlib.util.module_from_spec(spec)
loader.exec_module(coordinator_mod)
MultiAgentCoordinator = coordinator_mod.MultiAgentCoordinator
RLNegotiator = coordinator_mod.RLNegotiator

loader2 = importlib.machinery.SourceFileLoader(
    'meta_rl_refactor', 'src/meta_rl_refactor.py'
)
spec2 = importlib.util.spec_from_loader(loader2.name, loader2)
meta_mod = importlib.util.module_from_spec(spec2)
loader2.exec_module(meta_mod)
MetaRLRefactorAgent = meta_mod.MetaRLRefactorAgent


class TestMultiAgentCoordinator(unittest.IsolatedAsyncioTestCase):
    async def test_schedule_round(self):
        a1 = MetaRLRefactorAgent(epsilon=0.0)
        a2 = MetaRLRefactorAgent(epsilon=0.0)
        coord = MultiAgentCoordinator({'a1': a1, 'a2': a2})

        calls: list[tuple[str, str]] = []

        async def apply_fn(repo: str, action: str) -> None:
            calls.append((repo, action))

        def reward_fn(repo: str, action: str) -> float:
            return 1.0

        await coord.schedule_round(['r1', 'r2'], apply_fn=apply_fn, reward_fn=reward_fn)

        self.assertEqual(len(calls), 4)

        coord.train_agents()
        self.assertTrue(a1.q)
        self.assertTrue(a2.q)

    async def test_negotiation(self):
        a1 = MetaRLRefactorAgent(epsilon=0.0)
        a2 = MetaRLRefactorAgent(epsilon=0.0)
        neg = RLNegotiator(epsilon=0.0)
        coord = MultiAgentCoordinator({'a1': a1, 'a2': a2}, negotiator=neg)

        counts: list[str] = []

        async def apply_fn(repo: str, action: str) -> None:
            counts.append(repo)

        def reward_fn(repo: str, action: str) -> float:
            return 1.0

        await coord.schedule_round(['r1', 'r2'], apply_fn=apply_fn, reward_fn=reward_fn)
        self.assertEqual(len(counts), 2)


if __name__ == '__main__':
    unittest.main()
