import unittest
import importlib.machinery
import importlib.util
import sys
import numpy as np
import torch

loader_mm = importlib.machinery.SourceFileLoader('mm', 'src/multimodal_world_model.py')
spec_mm = importlib.util.spec_from_loader(loader_mm.name, loader_mm)
mm = importlib.util.module_from_spec(spec_mm)
sys.modules['mm'] = mm
sys.modules['asi.multimodal_world_model'] = mm
loader_mm.exec_module(mm)
MultiModalWorldModel = mm.MultiModalWorldModel
MultiModalWorldModelConfig = mm.MultiModalWorldModelConfig

loader_al = importlib.machinery.SourceFileLoader('al', 'src/auto_labeler.py')
spec_al = importlib.util.spec_from_loader(loader_al.name, loader_al)
al = importlib.util.module_from_spec(spec_al)
sys.modules['al'] = al
sys.modules['asi.auto_labeler'] = al
loader_al.exec_module(al)
AutoLabeler = al.AutoLabeler

loader_di = importlib.machinery.SourceFileLoader('di', 'src/data_ingest.py')
spec_di = importlib.util.spec_from_loader(loader_di.name, loader_di)
di = importlib.util.module_from_spec(spec_di)
sys.modules['di'] = di
sys.modules['asi.data_ingest'] = di
loader_di.exec_module(di)
auto_label_triples = di.auto_label_triples

class TestAutoLabeler(unittest.TestCase):
    def test_labeler_runs(self):
        cfg = MultiModalWorldModelConfig(vocab_size=10, img_channels=1, action_dim=2, embed_dim=4)
        model = MultiModalWorldModel(cfg)
        tok = lambda s: [ord(c) % cfg.vocab_size for c in s]
        labeler = AutoLabeler(model, tok)
        text = "hi"
        img = np.zeros((1, 8, 8), dtype=np.float32)
        lbl = labeler.label([(text, img, None)])
        self.assertEqual(len(lbl), 1)
        self.assertIsInstance(lbl[0], int)

    def test_auto_label_triples(self):
        cfg = MultiModalWorldModelConfig(vocab_size=10, img_channels=1, action_dim=2, embed_dim=4)
        model = MultiModalWorldModel(cfg)
        tok = lambda s: [ord(c) % cfg.vocab_size for c in s]
        labeler = AutoLabeler(model, tok)
        with unittest.mock.patch('builtins.open', unittest.mock.mock_open(read_data='hi')):
            import tempfile
            from pathlib import Path
            with tempfile.TemporaryDirectory() as root:
                t = Path(root)/'t.txt'; t.write_text('hi')
                i = Path(root)/'i.npy'; np.save(i, np.zeros((1,8,8), dtype=np.float32))
                a = Path(root)/'a.wav'; a.write_text('')
                labels = auto_label_triples([(t, i, a)], labeler)
                self.assertEqual(len(labels), 1)

if __name__ == '__main__':
    unittest.main()
