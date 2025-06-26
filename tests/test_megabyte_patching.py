import unittest
import torch

from asi.megabyte_patching import MegaBytePatching

class TestMegaBytePatching(unittest.TestCase):
    def test_patch_shape(self):
        module = MegaBytePatching(patch_size=4, dim=8)
        x = torch.randint(0, 256, (2, 10), dtype=torch.long)
        out = module(x)
        self.assertEqual(out.shape, (2, 3, 8))

if __name__ == '__main__':
    unittest.main()
