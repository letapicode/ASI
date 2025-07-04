import unittest
import torch
import importlib.machinery
import importlib.util
import types
import sys

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg

def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    sys.modules[name] = mod
    return mod

CausalReasoner = _load('asi.causal_reasoner', 'src/causal_reasoner.py').CausalReasoner
_load('asi.causal_graph_learner', 'src/causal_graph_learner.py')
wm = _load('asi.world_model_rl', 'src/world_model_rl.py')
RLBridgeConfig = wm.RLBridgeConfig
TransitionDataset = wm.TransitionDataset
train_world_model = wm.train_world_model


class TestCausalReasoner(unittest.TestCase):
    def test_plan(self):
        cfg = RLBridgeConfig(state_dim=2, action_dim=1, epochs=1, batch_size=1)
        transitions = []
        for _ in range(3):
            s = torch.randn(2)
            ns = s + 1
            transitions.append((s, 0, ns, 1.0))
        dataset = TransitionDataset(transitions)
        model = train_world_model(cfg, dataset)
        reasoner = CausalReasoner(model)
        reasoner.build_graph([(s, a, ns) for s, a, ns, _ in transitions])
        def policy(state: torch.Tensor) -> torch.Tensor:
            return torch.zeros((), dtype=torch.long)
        plan = reasoner.plan(policy, torch.zeros(2), steps=2)
        self.assertIn("edges", plan)
        self.assertEqual(len(plan["states"]), 2)


if __name__ == "__main__":
    unittest.main()
