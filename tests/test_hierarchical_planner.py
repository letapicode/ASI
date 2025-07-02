import unittest
import torch
from asi.graph_of_thought import GraphOfThought
from asi.hierarchical_planner import HierarchicalPlanner
from asi.world_model_rl import WorldModel, RLBridgeConfig

class TestHierarchicalPlanner(unittest.TestCase):
    def test_compose_plan(self):
        cfg = RLBridgeConfig(state_dim=1, action_dim=1, epochs=1)
        model = WorldModel(cfg)
        graph = GraphOfThought()
        graph.add_step("start", node_id=0)
        graph.add_step("goal", node_id=1)
        graph.connect(0, 1)
        planner = HierarchicalPlanner(graph, model, lambda s: torch.zeros((), dtype=torch.long))
        path, states, rewards = planner.compose_plan(0, lambda n: n.id == 1, torch.zeros(1))
        self.assertEqual(path, [0, 1])
        self.assertEqual(len(states), 2)
        self.assertEqual(len(rewards), 2)

if __name__ == "__main__":
    unittest.main()
