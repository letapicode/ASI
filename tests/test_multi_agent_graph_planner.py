import unittest
import asyncio
import importlib.machinery
import importlib.util
import sys
import types

asi_pkg = types.ModuleType('asi')
asi_pkg.__path__ = []
sys.modules.setdefault('asi', asi_pkg)

loader = importlib.machinery.SourceFileLoader('asi.multi_agent_coordinator', 'src/multi_agent_coordinator.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
coord_mod = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = coord_mod
loader.exec_module(coord_mod)
MultiAgentCoordinator = coord_mod.MultiAgentCoordinator

loader2 = importlib.machinery.SourceFileLoader('asi.meta_rl_refactor', 'src/meta_rl_refactor.py')
spec2 = importlib.util.spec_from_loader(loader2.name, loader2)
mrf_mod = importlib.util.module_from_spec(spec2)
sys.modules[loader2.name] = mrf_mod
loader2.exec_module(mrf_mod)
MetaRLRefactorAgent = mrf_mod.MetaRLRefactorAgent

loader3 = importlib.machinery.SourceFileLoader('asi.adaptive_planner', 'src/adaptive_planner.py')
spec3 = importlib.util.spec_from_loader(loader3.name, loader3)
ap_mod = importlib.util.module_from_spec(spec3)
sys.modules[loader3.name] = ap_mod
loader3.exec_module(ap_mod)
GraphOfThoughtPlanner = ap_mod.GraphOfThoughtPlanner

loader4 = importlib.machinery.SourceFileLoader('asi.multi_agent_graph_planner', 'src/multi_agent_graph_planner.py')
spec4 = importlib.util.spec_from_loader(loader4.name, loader4)
magp_mod = importlib.util.module_from_spec(spec4)
sys.modules[loader4.name] = magp_mod
sys.modules['asi.multi_agent_graph_planner'] = magp_mod
loader4.exec_module(magp_mod)
MultiAgentGraphPlanner = magp_mod.MultiAgentGraphPlanner


class TestMultiAgentGraphPlanner(unittest.TestCase):
    def test_plan(self):
        agents = {"a": MetaRLRefactorAgent(), "b": MetaRLRefactorAgent()}
        coord = MultiAgentCoordinator(agents)
        planner = GraphOfThoughtPlanner(lambda s: len(s))
        magp = MultiAgentGraphPlanner(coord, planner)
        repos = ["r1", "r2"]
        strategies = ["refactor foo", "replace bar"]
        graph = asyncio.run(magp.plan(repos, strategies))
        self.assertEqual(set(graph.keys()), set(repos))
        for edges in graph.values():
            self.assertGreaterEqual(len(edges), 1)


if __name__ == '__main__':
    unittest.main()
