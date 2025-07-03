import unittest
import torch

from asi.multi_stage_oversight import MultiStageOversight
from asi.formal_verifier import check_grad_norm


class TestMultiStageOversight(unittest.TestCase):
    def test_review(self):
        model = torch.nn.Linear(1, 2)
        ovs = MultiStageOversight(
            ["be kind"],
            "no bad",
            ["good", "bad"],
            checks=[lambda: check_grad_norm(model, 10.0)],
        )
        ok, _ = ovs.review("good")
        self.assertTrue(ok)
        bad, _ = ovs.review("bad")
        self.assertFalse(bad)


if __name__ == "__main__":
    unittest.main()
