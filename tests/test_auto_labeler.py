import unittest
import importlib.machinery
import importlib.util
import sys
import types
import numpy as np
import torch

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg
loader_lb = importlib.machinery.SourceFileLoader('asi.loihi_backend', 'src/loihi_backend.py')
spec_lb = importlib.util.spec_from_loader(loader_lb.name, loader_lb)
lb = importlib.util.module_from_spec(spec_lb)
lb.__package__ = 'asi'
sys.modules['asi.loihi_backend'] = lb
loader_lb.exec_module(lb)
loader_sl = importlib.machinery.SourceFileLoader('asi.spiking_layers', 'src/spiking_layers.py')
spec_sl = importlib.util.spec_from_loader(loader_sl.name, loader_sl)
sl = importlib.util.module_from_spec(spec_sl)
sl.__package__ = 'asi'
sys.modules['asi.spiking_layers'] = sl
loader_sl.exec_module(sl)

loader_mm = importlib.machinery.SourceFileLoader('asi.multimodal_world_model', 'src/multimodal_world_model.py')
spec_mm = importlib.util.spec_from_loader(loader_mm.name, loader_mm)
mm = importlib.util.module_from_spec(spec_mm)
mm.__package__ = 'asi'
sys.modules['asi.multimodal_world_model'] = mm
loader_mm.exec_module(mm)
MultiModalWorldModel = mm.MultiModalWorldModel
MultiModalWorldModelConfig = mm.MultiModalWorldModelConfig

loader_al = importlib.machinery.SourceFileLoader('asi.auto_labeler', 'src/auto_labeler.py')
spec_al = importlib.util.spec_from_loader(loader_al.name, loader_al)
al = importlib.util.module_from_spec(spec_al)
al.__package__ = 'asi'
sys.modules['asi.auto_labeler'] = al
loader_al.exec_module(al)
AutoLabeler = al.AutoLabeler

loader_di = importlib.machinery.SourceFileLoader('asi.data_ingest', 'src/data_ingest.py')
spec_di = importlib.util.spec_from_loader(loader_di.name, loader_di)
di = importlib.util.module_from_spec(spec_di)
di.__package__ = 'asi'
sys.modules['asi.data_ingest'] = di
loader_di.exec_module(di)
auto_label_triples = di.auto_label_triples

class TestAutoLabeler(unittest.TestCase):
    def test_labeler_runs(self):
        cfg = MultiModalWorldModelConfig(vocab_size=10, img_channels=1, action_dim=2, embed_dim=8)
        model = MultiModalWorldModel(cfg)
        tok = lambda s: [ord(c) % cfg.vocab_size for c in s]
        labeler = AutoLabeler(model, tok)
        text = "hi"
        img = np.zeros((1, 8, 8), dtype=np.float32)
        lbl = labeler.label([(text, img, None)])
        self.assertEqual(len(lbl), 1)
        self.assertIsInstance(lbl[0], int)

    def test_auto_label_triples(self):
        cfg = MultiModalWorldModelConfig(vocab_size=10, img_channels=1, action_dim=2, embed_dim=8)
        model = MultiModalWorldModel(cfg)
        tok = lambda s: [ord(c) % cfg.vocab_size for c in s]
        labeler = AutoLabeler(model, tok)
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as root:
            t = Path(root)/'t.txt'; t.write_text('hi')
            i = Path(root)/'i.npy'; np.save(i, np.zeros((1,8,8), dtype=np.float32))
            a = Path(root)/'a.wav'; a.write_text('')
            labels = auto_label_triples([(str(t), str(i), str(a))], labeler)
            self.assertEqual(len(labels), 1)

if __name__ == '__main__':
    unittest.main()
