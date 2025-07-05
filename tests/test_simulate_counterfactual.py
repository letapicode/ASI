import unittest
import importlib.machinery
import importlib.util
import types
import sys
import torch

# prepare asi package
pkg = types.ModuleType('asi')
sys.modules.setdefault('asi', pkg)

def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    sys.modules[name] = mod
    return mod

CausalGraphLearner = _load('asi.causal_graph_learner', 'src/causal_graph_learner.py').CausalGraphLearner
wm_mod = _load('asi.world_model_rl', 'src/world_model_rl.py')
RLBridgeConfig = wm_mod.RLBridgeConfig
TransitionDataset = wm_mod.TransitionDataset
train_world_model = wm_mod.train_world_model
simulate_counterfactual = wm_mod.simulate_counterfactual


class TestSimulateCounterfactual(unittest.TestCase):
    def test_counterfactual(self):
        cfg = RLBridgeConfig(state_dim=2, action_dim=1, epochs=1, batch_size=1)
        data = []
        for i in range(2):
            s = torch.tensor([float(i), float(i)])
            ns = s + 1
            data.append((s, 0, ns, 0.0))
        ds = TransitionDataset(data)
        learner = CausalGraphLearner()
        model = train_world_model(cfg, ds, learner=learner)
        s = torch.tensor([0.0, 0.0])
        a = torch.tensor(0)
        ns, r = simulate_counterfactual(model, learner, s, a, {0: 1.0})
        self.assertEqual(ns.shape[0], 2)
        self.assertIsInstance(r.item(), float)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
