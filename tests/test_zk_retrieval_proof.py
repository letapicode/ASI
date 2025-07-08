import unittest
import numpy as np
from asi.zk_retrieval_proof import ZKRetrievalProof


class TestZKRetrievalProof(unittest.TestCase):
    def test_generate_verify(self):
        vecs = np.random.randn(2, 3).astype(np.float32)
        meta = ["a", "b"]
        proof = ZKRetrievalProof.generate(vecs, meta)
        self.assertTrue(proof.verify(vecs, meta))
        self.assertFalse(proof.verify(vecs + 1, meta))


if __name__ == "__main__":
    unittest.main()
