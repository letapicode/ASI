import unittest
import torch
from asi.secure_federated_learner import SecureFederatedLearner

class TestSecureFederatedLearner(unittest.TestCase):
    def test_aggregate(self):
        learner = SecureFederatedLearner(key=1)
        grads = [learner.encrypt(torch.ones(2)), learner.encrypt(torch.zeros(2))]
        agg = learner.aggregate([learner.decrypt(g) for g in grads])
        self.assertTrue(torch.allclose(agg, torch.tensor([0.5, 0.5])))

    def test_proof_roundtrip(self):
        learner = SecureFederatedLearner(key=2)
        enc, proof = learner.encrypt(torch.ones(3), with_proof=True)
        dec = learner.decrypt(enc, proof)
        self.assertTrue(torch.allclose(dec, torch.ones(3)))

if __name__ == '__main__':
    unittest.main()
