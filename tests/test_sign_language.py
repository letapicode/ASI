import importlib.machinery
import importlib.util
import sys
import types
import unittest
import numpy as np
import torch

loader_sign = importlib.machinery.SourceFileLoader('sign', 'src/sign_language.py')
spec_sign = importlib.util.spec_from_loader(loader_sign.name, loader_sign)
sign = importlib.util.module_from_spec(spec_sign)
sign.__package__ = 'asi'
loader_sign.exec_module(sign)
SignLanguageRecognizer = sign.SignLanguageRecognizer

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg
sys.modules['asi.sign_language'] = sign
setattr(pkg, 'sign_language', sign)
loader_qmm = importlib.machinery.SourceFileLoader('qmm', 'src/quantum_multimodal_retrieval.py')
spec_qmm = importlib.util.spec_from_loader(loader_qmm.name, loader_qmm)
qmm = importlib.util.module_from_spec(spec_qmm)
qmm.__package__ = 'asi'
loader_qr = importlib.machinery.SourceFileLoader('qr', 'src/quantum_retrieval.py')
spec_qr = importlib.util.spec_from_loader(loader_qr.name, loader_qr)
qr = importlib.util.module_from_spec(spec_qr)
qr.__package__ = 'asi'
loader_qr.exec_module(qr)
sys.modules['asi.quantum_retrieval'] = qr
setattr(pkg, 'quantum_retrieval', qr)
loader_qmm.exec_module(qmm)
sys.modules['asi.quantum_multimodal_retrieval'] = qmm
setattr(pkg, 'quantum_multimodal_retrieval', qmm)

loader_cmf = importlib.machinery.SourceFileLoader('cmf', 'src/cross_modal_fusion.py')
spec_cmf = importlib.util.spec_from_loader(loader_cmf.name, loader_cmf)
cmf = importlib.util.module_from_spec(spec_cmf)
cmf.__package__ = 'asi'
sys.modules['asi.cross_modal_fusion'] = cmf
loader_cmf.exec_module(cmf)
setattr(pkg, 'cross_modal_fusion', cmf)

loader_hm = importlib.machinery.SourceFileLoader('hm', 'src/hierarchical_memory.py')
spec_hm = importlib.util.spec_from_loader(loader_hm.name, loader_hm)
hm = importlib.util.module_from_spec(spec_hm)
hm.__package__ = 'asi'
loader_hm.exec_module(hm)
sys.modules['asi.hierarchical_memory'] = hm
setattr(pkg, 'hierarchical_memory', hm)
CrossModalFusionConfig = cmf.CrossModalFusionConfig
CrossModalFusion = cmf.CrossModalFusion
MultiModalDataset = cmf.MultiModalDataset
encode_all = cmf.encode_all

from asi.hierarchical_memory import HierarchicalMemory

def tok(t: str):
    return [ord(c) % 50 for c in t]

class TestSignLanguage(unittest.TestCase):
    def test_recognizer(self):
        rec = SignLanguageRecognizer()
        vid = np.ones((2, 2, 2, 3), dtype=np.float32)
        self.assertEqual(rec.recognize(vid), "hello")

    def test_recognizer_thanks(self):
        rec = SignLanguageRecognizer()
        vid = np.ones((2, 2, 2, 3), dtype=np.float32) * 0.5
        self.assertEqual(rec.recognize(vid), "thanks")

    def test_recognizer_unknown(self):
        rec = SignLanguageRecognizer()
        vid = np.zeros((2, 2, 2, 3), dtype=np.float32)
        self.assertEqual(rec.recognize(vid), "")

    def test_retrieval_with_sign(self):
        cfg = CrossModalFusionConfig(vocab_size=50, text_dim=8, img_channels=3, audio_channels=1, latent_dim=4)
        model = CrossModalFusion(cfg)
        data = [
            ("a", torch.randn(3,16,16), torch.randn(1,32)),
            ("b", torch.randn(3,16,16), torch.randn(1,32)),
        ]
        ds = MultiModalDataset(data, tok)
        videos = [np.ones((1,1,3), dtype=np.float32) for _ in range(len(ds))]
        mem = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10)
        t,i,a,s = encode_all(model, ds, batch_size=1, memory=mem, include_sign=True, sign_videos=videos)
        q = (t[0] + i[0] + a[0] + s[0]) / 4.0
        out, meta = mem.search(q, k=1)
        self.assertEqual(meta[0], 0)

if __name__ == '__main__':
    unittest.main()
