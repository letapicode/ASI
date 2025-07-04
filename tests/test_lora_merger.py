import unittest
import importlib.machinery
import importlib.util
import types
import sys
import torch
import tempfile

src_pkg = types.ModuleType('src')
src_pkg.__path__ = ['src']
src_pkg.__spec__ = importlib.machinery.ModuleSpec('src', None, is_package=True)
sys.modules['src'] = src_pkg
loader = importlib.machinery.SourceFileLoader('src.lora_merger', 'src/lora_merger.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
mod = importlib.util.module_from_spec(spec)
mod.__package__ = 'src'
loader.exec_module(mod)
merge_adapters = mod.merge_adapters

class TestLoRAMerger(unittest.TestCase):
    def test_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            p1 = f"{tmp}/a.pt"
            p2 = f"{tmp}/b.pt"
            torch.save({'w': torch.ones(1)}, p1)
            torch.save({'w': torch.zeros(1)}, p2)
            merged = merge_adapters(None, [p1, p2], [0.5, 0.5])
            self.assertAlmostEqual(float(merged['w']), 0.5)

if __name__ == '__main__':
    unittest.main()
