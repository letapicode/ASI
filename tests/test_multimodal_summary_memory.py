import importlib.machinery
import importlib.util
import types
import sys
import unittest
import torch

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg

# load required modules
loader_qm = importlib.machinery.SourceFileLoader('qm', 'src/quantum_multimodal_retrieval.py')
spec_qm = importlib.util.spec_from_loader(loader_qm.name, loader_qm)
qm = importlib.util.module_from_spec(spec_qm)
qm.__package__ = 'asi'
loader_qm.exec_module(qm)
sys.modules['asi.quantum_multimodal_retrieval'] = qm
setattr(pkg, 'quantum_multimodal_retrieval', qm)

loader_sl = importlib.machinery.SourceFileLoader('sl', 'src/sign_language.py')
spec_sl = importlib.util.spec_from_loader(loader_sl.name, loader_sl)
sl = importlib.util.module_from_spec(spec_sl)
sl.__package__ = 'asi'
loader_sl.exec_module(sl)
sys.modules['asi.sign_language'] = sl
setattr(pkg, 'sign_language', sl)

# load cross-modal and memory modules
loader_cmf = importlib.machinery.SourceFileLoader('cmf', 'src/cross_modal_fusion.py')
spec_cmf = importlib.util.spec_from_loader(loader_cmf.name, loader_cmf)
cmf = importlib.util.module_from_spec(spec_cmf)
cmf.__package__ = 'asi'
loader_cmf.exec_module(cmf)
sys.modules['asi.cross_modal_fusion'] = cmf
setattr(pkg, 'cross_modal_fusion', cmf)

loader_hm = importlib.machinery.SourceFileLoader('hm', 'src/hierarchical_memory.py')
spec_hm = importlib.util.spec_from_loader(loader_hm.name, loader_hm)
hm = importlib.util.module_from_spec(spec_hm)
hm.__package__ = 'asi'
loader_hm.exec_module(hm)
sys.modules['asi.hierarchical_memory'] = hm
setattr(pkg, 'hierarchical_memory', hm)

loader_bs = importlib.machinery.SourceFileLoader('bs', 'src/summarizing_memory_base.py')
spec_bs = importlib.util.spec_from_loader(loader_bs.name, loader_bs)
bs = importlib.util.module_from_spec(spec_bs)
bs.__package__ = 'asi'
loader_bs.exec_module(bs)
sys.modules['asi.summarizing_memory_base'] = bs
setattr(pkg, 'summarizing_memory_base', bs)

loader_mm = importlib.machinery.SourceFileLoader('mm', 'src/multimodal_summary_memory.py')
spec_mm = importlib.util.spec_from_loader(loader_mm.name, loader_mm)
mm = importlib.util.module_from_spec(spec_mm)
mm.__package__ = 'asi'
loader_mm.exec_module(mm)
sys.modules['asi.multimodal_summary_memory'] = mm
setattr(pkg, 'multimodal_summary_memory', mm)

CrossModalFusionConfig = cmf.CrossModalFusionConfig
CrossModalFusion = cmf.CrossModalFusion
MultiModalDataset = cmf.MultiModalDataset
encode_all = cmf.encode_all
MultiModalSummaryMemory = mm.MultiModalSummaryMemory


def tok(t: str):
    return [ord(c) % 50 for c in t]


class DummySummarizer:
    def summarize(self, x):
        return 'sum'

    def expand(self, text):
        return torch.ones(4)


class TestMultiModalSummaryMemory(unittest.TestCase):
    def test_retrieval_accuracy(self):
        cfg = CrossModalFusionConfig(
            vocab_size=50,
            text_dim=8,
            img_channels=3,
            audio_channels=1,
            latent_dim=4,
            bci_channels=1,
        )
        model = CrossModalFusion(cfg)
        triples = [
            ("foo", torch.randn(3, 16, 16), torch.randn(1, 32)),
            ("bar", torch.randn(3, 16, 16), torch.randn(1, 32)),
        ]
        ds = MultiModalDataset(triples, tok)
        mem = MultiModalSummaryMemory(
            dim=4,
            compressed_dim=2,
            capacity=10,
            image_summarizer=DummySummarizer(),
            audio_summarizer=DummySummarizer(),
        )
        mem.add_dataset(model, ds, batch_size=1)
        t_vecs, i_vecs, a_vecs, _ = encode_all(model, ds, batch_size=1)
        img_v = mem.image_summarizer.expand('sum')
        aud_v = mem.audio_summarizer.expand('sum')
        for idx in range(len(ds)):
            q = (t_vecs[idx] + img_v + aud_v) / 3.0
            out, meta = mem.search(q, k=2)
            ids = [m['id'] for m in meta]
            self.assertIn(idx, ids)


if __name__ == '__main__':
    unittest.main()
