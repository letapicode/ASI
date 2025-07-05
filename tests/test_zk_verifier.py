import unittest
import importlib.machinery
import importlib.util
import torch

loader = importlib.machinery.SourceFileLoader('zk', 'src/zk_verifier.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
zk = importlib.util.module_from_spec(spec)
loader.exec_module(zk)
ZKVerifier = zk.ZKVerifier


class TestZKVerifier(unittest.TestCase):
    def test_roundtrip(self):
        v = ZKVerifier()
        g = torch.tensor([1.0, 2.0])
        p = v.generate_proof(g)
        self.assertTrue(v.verify_proof(g, p))

    def test_bad_proof(self):
        v = ZKVerifier()
        g = torch.tensor([0.0])
        p = v.generate_proof(g)
        self.assertFalse(v.verify_proof(g + 1, p))


if __name__ == '__main__':
    unittest.main()

