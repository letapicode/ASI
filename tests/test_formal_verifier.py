import unittest
import torch

from asi.formal_verifier import (
    check_grad_norm,
    check_output_bounds,
    verify_model,
    VerificationResult,
)


class TestFormalVerifier(unittest.TestCase):
    def test_check_grad_norm(self):
        model = torch.nn.Linear(2, 1)
        x = torch.ones(1, 2)
        model(x).sum().backward()
        ok, _ = check_grad_norm(model, max_norm=100.0)
        self.assertTrue(ok)
        ok, _ = check_grad_norm(model, max_norm=0.0)
        self.assertFalse(ok)

    def test_check_output_bounds(self):
        output = torch.tensor([0.5, -0.2])
        self.assertTrue(check_output_bounds(output, 1.0)[0])
        self.assertFalse(check_output_bounds(output, 0.3)[0])

    def test_verify_model(self):
        checks = [lambda: (True, "ok1"), lambda: (True, "ok2")]
        res = verify_model(torch.nn.Linear(1, 1), checks)
        self.assertIsInstance(res, VerificationResult)
        self.assertTrue(res.passed)
        res_fail = verify_model(torch.nn.Linear(1, 1), [lambda: (False, "bad")])
        self.assertFalse(res_fail.passed)
        self.assertEqual(res_fail.messages[0], "bad")


if __name__ == "__main__":
    unittest.main()
