import unittest
import torch
from asi.topk_sparse_attention import topk_sparse_attention

class TestTopkSparseAttention(unittest.TestCase):
    def test_shape_and_equivalence(self):
        q = torch.randn(2, 3, 4)
        k = torch.randn(2, 5, 4)
        v = torch.randn(2, 5, 4)
        # k_top less than sequence length
        out = topk_sparse_attention(q, k, v, k_top=2)
        self.assertEqual(out.shape, (2, 3, 4))
        # k_top equal to seq_k should match full attention
        full_scores = torch.matmul(q, k.transpose(-1, -2)) / (4 ** 0.5)
        full_attn = torch.softmax(full_scores, dim=-1)
        full_out = torch.matmul(full_attn, v)
        out_full = topk_sparse_attention(q, k, v, k_top=k.size(1))
        self.assertTrue(torch.allclose(out_full, full_out, atol=1e-5, rtol=1e-5))

    def test_invalid_k_top(self):
        q = torch.randn(1, 2, 4)
        k = torch.randn(1, 3, 4)
        v = torch.randn(1, 3, 4)
        with self.assertRaises(ValueError):
            topk_sparse_attention(q, k, v, k_top=4)

if __name__ == '__main__':
    unittest.main()
