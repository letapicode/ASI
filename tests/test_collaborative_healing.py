import unittest
import importlib.machinery
import importlib.util
import types
import sys

src_pkg = types.ModuleType('src')
sys.modules['src'] = src_pkg
import asyncio

loader = importlib.machinery.SourceFileLoader('src.graph_of_thought', 'src/graph_of_thought.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
got = importlib.util.module_from_spec(spec)
got.__package__ = 'src'
sys.modules['src.graph_of_thought'] = got
loader.exec_module(got)

loader = importlib.machinery.SourceFileLoader('src.multi_agent_coordinator', 'src/multi_agent_coordinator.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
mac = importlib.util.module_from_spec(spec)
mac.__package__ = 'src'
sys.modules['src.multi_agent_coordinator'] = mac
loader.exec_module(mac)
MultiAgentCoordinator = mac.MultiAgentCoordinator

loader = importlib.machinery.SourceFileLoader('src.collaborative_healing', 'src/collaborative_healing.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
ch = importlib.util.module_from_spec(spec)
ch.__package__ = 'src'
sys.modules['src.collaborative_healing'] = ch
loader.exec_module(ch)
CollaborativeHealingLoop = ch.CollaborativeHealingLoop


class DummyAgent:
    def select_action(self, repo: str) -> str:
        return 'patch'

    def update(self, repo: str, action: str, reward: float, context: str) -> None:
        pass

    def train(self, log):
        pass


class TestCollaborativeHealing(unittest.TestCase):
    def test_loop(self):
        coord = MultiAgentCoordinator({'a': DummyAgent()})
        loop = CollaborativeHealingLoop(coord)
        node = loop.report_anomaly('comp', 'fail')
        self.assertEqual(node, 0)
        asyncio.run(loop.run(['comp']))


if __name__ == '__main__':
    unittest.main()
