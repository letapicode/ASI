import unittest
import importlib.machinery
import importlib.util
import types
import sys

try:
    import torch
except Exception:  # pragma: no cover - torch optional
    torch = None

class DummyAnalog:
    def __init__(self):
        self.calls = 0
    def matmul(self, a, b, noise=0.0):
        self.calls += 1
        return a @ b

def load_modules(has_analog=True):
    pkg = types.ModuleType('asi')
    pkg.__path__ = ['src']
    sys.modules['asi'] = pkg

    loader_ab = importlib.machinery.SourceFileLoader('asi.analog_backend', 'src/analog_backend.py')
    spec_ab = importlib.util.spec_from_loader(loader_ab.name, loader_ab)
    ab = importlib.util.module_from_spec(spec_ab)
    ab.__package__ = 'asi'
    sys.modules['asi.analog_backend'] = ab
    loader_ab.exec_module(ab)

    dummy = DummyAnalog()
    ab._HAS_ANALOG = has_analog
    ab.analogsim = dummy

    lora_stub = types.ModuleType('asi.lora_quant')
    lora_stub.apply_quant_lora = lambda *a, **k: None
    sys.modules['asi.lora_quant'] = lora_stub

    for mod_name in ['spiking_layers', 'telemetry']:
        loader = importlib.machinery.SourceFileLoader(f'asi.{mod_name}', f'src/{mod_name}.py')
        spec = importlib.util.spec_from_loader(loader.name, loader)
        m = importlib.util.module_from_spec(spec)
        m.__package__ = 'asi'
        sys.modules[f'asi.{mod_name}'] = m
        loader.exec_module(m)

    loader_mm = importlib.machinery.SourceFileLoader('asi.multimodal_world_model', 'src/multimodal_world_model.py')
    spec_mm = importlib.util.spec_from_loader(loader_mm.name, loader_mm)
    mm = importlib.util.module_from_spec(spec_mm)
    mm.__package__ = 'asi'
    sys.modules['asi.multimodal_world_model'] = mm
    loader_mm.exec_module(mm)
    return mm, ab, dummy

@unittest.skipIf(torch is None, "torch not available")
class TestAnalogForward(unittest.TestCase):
    def test_forward_with_analog(self):
        mm, ab, dummy = load_modules(True)
        cfg = mm.MultiModalWorldModelConfig(vocab_size=5, img_channels=1, action_dim=2, embed_dim=4, use_analog=True)
        model = mm.MultiModalWorldModel(cfg)
        t = torch.zeros(1, 1, dtype=torch.long)
        img = torch.zeros(1, 1, 4, 4)
        a = torch.zeros(1, dtype=torch.long)
        model(t, img, a)
        self.assertGreater(dummy.calls, 0)

    def test_cpu_fallback(self):
        mm, ab, dummy = load_modules(False)
        cfg = mm.MultiModalWorldModelConfig(vocab_size=5, img_channels=1, action_dim=2, embed_dim=4, use_analog=True)
        model = mm.MultiModalWorldModel(cfg)
        t = torch.zeros(1, 1, dtype=torch.long)
        img = torch.zeros(1, 1, 4, 4)
        a = torch.zeros(1, dtype=torch.long)
        model(t, img, a)
        self.assertEqual(dummy.calls, 0)

if __name__ == '__main__':
    unittest.main()
