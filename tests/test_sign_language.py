import importlib.machinery
import importlib.util
import sys
import types
import unittest
import numpy as np
import torch
from unittest.mock import patch

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg


def load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'asi'
    loader.exec_module(mod)
    sys.modules[name] = mod
    setattr(pkg, name.split('.')[-1], mod)
    return mod

cmf = load('asi.cross_modal_fusion', 'src/cross_modal_fusion.py')
clm = load('asi.cross_lingual_memory', 'src/cross_lingual_memory.py')
sl = load('asi.sign_language', 'src/sign_language.py')

SignLanguageRecognizer = sl.SignLanguageRecognizer
CrossLingualMemory = clm.CrossLingualMemory
CrossModalFusionConfig = cmf.CrossModalFusionConfig
CrossModalFusion = cmf.CrossModalFusion
MultiModalDataset = cmf.MultiModalDataset


class TestSignLanguage(unittest.TestCase):
    def test_recognize_and_retrieve(self):
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        rec = SignLanguageRecognizer({'hi': ref})
        with patch.object(rec, 'encode', return_value=ref):
            mem = CrossLingualMemory(dim=3, compressed_dim=2, capacity=10, sign_recognizer=rec)
            sign = torch.from_numpy(ref)
            mem.add_modalities(sign=sign.unsqueeze(0), metadata=[0])
            vecs, meta = mem.search_text('hi', k=1)
            self.assertEqual(len(meta), 1)
            self.assertEqual(meta[0], 0)

    def test_encode_all_store_sign(self):
        cfg = CrossModalFusionConfig(vocab_size=10, text_dim=3, img_channels=1, audio_channels=1, latent_dim=3)
        model = CrossModalFusion(cfg)
        data = [('hi', torch.zeros(1,1,1), torch.zeros(1,1), np.array([1.0,0.0,0.0], dtype=np.float32))]
        ds = MultiModalDataset(data, lambda x: [0])
        rec = SignLanguageRecognizer({'hi': np.array([1.0,0.0,0.0], dtype=np.float32)})
        with patch.object(rec, 'encode', return_value=np.array([1.0,0.0,0.0], dtype=np.float32)):
            mem = CrossLingualMemory(dim=3, compressed_dim=2, capacity=10, sign_recognizer=rec)
            cmf.encode_all(model, ds, batch_size=1, memory=mem, sign_recognizer=rec)
            vecs, meta = mem.search_text('hi', k=1)
            self.assertTrue(meta)


if __name__ == '__main__':
    unittest.main()
