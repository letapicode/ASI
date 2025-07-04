import unittest
import importlib.machinery
import importlib.util
import types
import sys
import torch

src_pkg = types.ModuleType('src')
src_pkg.__path__ = ['src']
src_pkg.__spec__ = importlib.machinery.ModuleSpec('src', None, is_package=True)
sys.modules['src'] = src_pkg
loader = importlib.machinery.SourceFileLoader('src.gradient_patch_editor', 'src/gradient_patch_editor.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
gpe = importlib.util.module_from_spec(spec)
gpe.__package__ = 'src'
sys.modules['src.gradient_patch_editor'] = gpe
loader.exec_module(gpe)

loader = importlib.machinery.SourceFileLoader('src.semantic_drift_detector', 'src/semantic_drift_detector.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
sdd = importlib.util.module_from_spec(spec)
sdd.__package__ = 'src'
sys.modules['src.semantic_drift_detector'] = sdd
loader.exec_module(sdd)
SemanticDriftDetector = sdd.SemanticDriftDetector
SemanticDriftDetector = sdd.SemanticDriftDetector

loader2 = importlib.machinery.SourceFileLoader('src.world_model_debugger', 'src/world_model_debugger.py')
spec2 = importlib.util.spec_from_loader(loader2.name, loader2)
wd = importlib.util.module_from_spec(spec2)
wd.__package__ = 'src'
sys.modules['src.world_model_debugger'] = wd
loader2.exec_module(wd)
WorldModelDebugger = wd.WorldModelDebugger

class ToyModel(torch.nn.Module):
    def forward(self, s, a):
        return s + a, None

class TestSemanticDriftDetector(unittest.TestCase):
    def test_update(self):
        det = SemanticDriftDetector()
        x1 = torch.randn(2, 3)
        x2 = torch.randn(2, 3)
        d1 = det.update(x1)
        d2 = det.update(x2)
        self.assertEqual(d1, 0.0)
        self.assertGreaterEqual(d2, 0.0)

    def test_debugger_integration(self):
        det = SemanticDriftDetector()
        dbg = WorldModelDebugger(ToyModel(), drift_detector=det)
        s = torch.randn(1, 2)
        a = torch.randn(1, 2)
        t = s + a
        loss = dbg.check(s, a, t)
        self.assertAlmostEqual(loss, 0.0, places=5)
        self.assertIsNotNone(det.last_drift())

if __name__ == '__main__':
    unittest.main()
