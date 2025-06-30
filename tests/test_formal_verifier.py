import unittest
import importlib.machinery
import importlib.util
import sys
import torch

loader = importlib.machinery.SourceFileLoader('fv', 'src/formal_verifier.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
fv = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = fv
loader.exec_module(fv)
check_grad_norm = fv.check_grad_norm
check_output_bounds = fv.check_output_bounds
verify_model = fv.verify_model
VerificationResult = fv.VerificationResult


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)


def make_model_with_grad(value: float) -> DummyModel:
    model = DummyModel()
    for p in model.parameters():
        p.grad = torch.full_like(p.data, value)
    return model


class TestFormalVerifier(unittest.TestCase):
    def test_check_grad_norm(self):
        model = make_model_with_grad(0.5)
        ok, _ = check_grad_norm(model, 1.0)
        self.assertTrue(ok)
        model = make_model_with_grad(2.0)
        ok, _ = check_grad_norm(model, 1.0)
        self.assertFalse(ok)

    def test_check_output_bounds(self):
        ok, _ = check_output_bounds(torch.tensor([0.1, -0.1]), 1.0)
        self.assertTrue(ok)
        ok, _ = check_output_bounds(torch.tensor([2.0]), 1.0)
        self.assertFalse(ok)

    def test_verify_model(self):
        calls = []

        def c1():
            calls.append("c1")
            return True, "ok1"

        def c2():
            calls.append("c2")
            return False, "fail"

        def c3():
            calls.append("c3")
            return True, "ok3"

        result = verify_model(DummyModel(), [c1, c2, c3])
        self.assertIsInstance(result, VerificationResult)
        self.assertFalse(result.passed)
        self.assertEqual(result.messages, ["ok1", "fail"])
        self.assertEqual(calls, ["c1", "c2"])


if __name__ == "__main__":
    unittest.main()
