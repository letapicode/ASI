import unittest
import torch
from asi.proofs import ZKGradientProof


class TestZKGradientProof(unittest.TestCase):
    def test_generate_verify(self):
        grad = torch.randn(3)
        proof = ZKGradientProof.generate(grad)
        self.assertTrue(proof.verify(grad))
        self.assertFalse(proof.verify(grad + 1))


if __name__ == "__main__":
    unittest.main()
