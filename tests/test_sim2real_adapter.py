import unittest
import importlib.machinery
import importlib.util
import sys
import torch

loader = importlib.machinery.SourceFileLoader('s2r', 'src/sim2real_adapter.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
s2r = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = s2r
loader.exec_module(s2r)
learn_env_params = s2r.learn_env_params
apply_correction = s2r.apply_correction
Sim2RealParams = s2r.Sim2RealParams


class TestSim2RealAdapter(unittest.TestCase):
    def test_learn_and_apply(self):
        logs = [
            (torch.zeros(2), torch.tensor([1.0, 0.5])),
            (torch.ones(2), torch.tensor([2.0, 1.5])),
        ]
        params = learn_env_params(logs)
        self.assertIsInstance(params, Sim2RealParams)
        self.assertTrue(torch.allclose(params.bias, torch.tensor([1.0, 0.5])))

        transitions = [(torch.zeros(2), 0, torch.zeros(2), 0.0)]
        corrected = apply_correction(transitions, params)
        s, a, ns, r = corrected[0]
        self.assertTrue(torch.allclose(s, torch.tensor([1.0, 0.5])))
        self.assertTrue(torch.allclose(ns, torch.tensor([1.0, 0.5])))


if __name__ == "__main__":
    unittest.main()
