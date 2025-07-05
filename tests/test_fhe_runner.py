import unittest
import torch
import tenseal as ts

from asi.fhe_runner import run_fhe


def add_one(x):
    return x + 1


class TestFHERunner(unittest.TestCase):
    def test_run_fhe(self):
        ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
        )
        ctx.generate_galois_keys()
        ctx.global_scale = 2**40
        inp = torch.tensor([1.0, 2.0])
        out = run_fhe(add_one, inp, ctx)
        exp = torch.tensor([2.0, 3.0])
        self.assertTrue(torch.allclose(out, exp, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
