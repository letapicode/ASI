import unittest
import torch
import importlib

import src.flash_attention3 as fa3

class TestFlashAttention3(unittest.TestCase):
    def test_fallback(self):
        original = fa3._HAS_FLASH3
        fa3._HAS_FLASH3 = False
        try:
            q = torch.randn(1, 4, 8)
            k = torch.randn(1, 4, 8)
            v = torch.randn(1, 4, 8)
            out = fa3.flash_attention_3(q, k, v)
            self.assertEqual(out.shape, q.shape)
        finally:
            fa3._HAS_FLASH3 = original

if __name__ == "__main__":
    unittest.main()
