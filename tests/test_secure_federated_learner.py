import unittest
import importlib.machinery
import importlib.util
import types
import sys
import torch

pkg = types.ModuleType("src")
pkg.__path__ = ["src"]
pkg.__spec__ = importlib.machinery.ModuleSpec("src", None, is_package=True)
sys.modules["src"] = pkg

loader = importlib.machinery.SourceFileLoader("src.secure_federated_learner", "src/secure_federated_learner.py")
spec = importlib.util.spec_from_loader(loader.name, loader)
mod = importlib.util.module_from_spec(spec)
mod.__package__ = "src"
sys.modules[loader.name] = mod
loader.exec_module(mod)
SecureFederatedLearner = mod.SecureFederatedLearner

class TestSecureFederatedLearner(unittest.TestCase):
    def test_aggregate(self):
        learner = SecureFederatedLearner(key=1)
        grads = [learner.encrypt(torch.ones(2)), learner.encrypt(torch.zeros(2))]
        agg = learner.aggregate([learner.decrypt(g) for g in grads])
        self.assertTrue(torch.allclose(agg, torch.tensor([0.5, 0.5])))

    def test_proof_required(self):
        learner = SecureFederatedLearner(key=1, require_proof=True)
        g = torch.ones(2)
        enc = learner.encrypt(g)
        proof = learner.zk.generate_proof(g)
        dec = learner.decrypt(enc)
        agg = learner.aggregate([dec], proofs=[proof])
        self.assertTrue(torch.allclose(agg, g))

if __name__ == '__main__':
    unittest.main()
