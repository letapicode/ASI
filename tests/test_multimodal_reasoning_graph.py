import unittest
import importlib.machinery
import importlib.util
import types
import sys
import tempfile
try:
    import torch
except Exception:  # pragma: no cover - torch optional
    raise unittest.SkipTest("torch not available")

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg

def load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'asi'
    loader.exec_module(mod)
    sys.modules[name] = mod
    setattr(pkg, name.split('.')[-1], mod)
    return mod

cmf = load('asi.cross_modal_fusion', 'src/cross_modal_fusion.py')
clg_mem = load('asi.context_summary_memory', 'src/context_summary_memory.py')
clg_data = load('asi.data_ingest', 'src/data_ingest.py')
goth = load('asi.graph_of_thought', 'src/graph_of_thought.py')
clg = load('asi.cross_lingual_graph', 'src/cross_lingual_graph.py')
rh = load('asi.reasoning_history', 'src/reasoning_history.py')
hm = load('asi.hierarchical_memory', 'src/hierarchical_memory.py')

CrossModalFusionConfig = cmf.CrossModalFusionConfig
CrossModalFusion = cmf.CrossModalFusion
embed_modalities = cmf.embed_modalities
CrossLingualReasoningGraph = clg.CrossLingualReasoningGraph
ReasoningHistoryLogger = rh.ReasoningHistoryLogger
HierarchicalMemory = hm.HierarchicalMemory


def simple_tokenizer(text: str):
    return [ord(c) % 50 for c in text]


class TestMultimodalReasoningGraph(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        cfg = CrossModalFusionConfig(
            vocab_size=50,
            text_dim=8,
            img_channels=3,
            audio_channels=1,
            latent_dim=4,
        )
        self.model = CrossModalFusion(cfg)
        self.graph = CrossLingualReasoningGraph()
        self.mem = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10)
        self.logger = ReasoningHistoryLogger()

    def test_multimodal_workflow(self):
        img0 = torch.randn(3, 8, 8)
        aud0 = torch.randn(1, 16)
        t0, i0, a0 = embed_modalities(self.model, simple_tokenizer, 'a', img0, aud0)
        n0 = self.graph.add_step('a', image_embed=i0, audio_embed=a0)
        fused0 = (torch.tensor(t0) + torch.tensor(i0) + torch.tensor(a0)) / 3.0
        self.mem.add(fused0, metadata=[n0])
        self.logger.log({'summary': 'a', 'image_vec': i0, 'audio_vec': a0})

        img1 = torch.randn(3, 8, 8)
        aud1 = torch.randn(1, 16)
        t1, i1, a1 = embed_modalities(self.model, simple_tokenizer, 'b', img1, aud1)
        n1 = self.graph.add_step('b', image_embed=i1, audio_embed=a1)
        fused1 = (torch.tensor(t1) + torch.tensor(i1) + torch.tensor(a1)) / 3.0
        self.mem.add(fused1, metadata=[n1])
        self.logger.log({'summary': 'b', 'image_vec': i1, 'audio_vec': a1})

        q_t, q_i, q_a = embed_modalities(self.model, simple_tokenizer, 'a', img0, aud0)
        query = (torch.tensor(q_t) + torch.tensor(q_i) + torch.tensor(q_a)) / 3.0
        _vec, meta = self.mem.search(query, k=1)
        self.assertEqual(meta[0], n0)
        self.assertIn('image_vec', self.graph.nodes[n0].metadata)

        with tempfile.NamedTemporaryFile('w+', delete=False) as f:
            self.logger.save(f.name)
            f.seek(0)
            loaded = ReasoningHistoryLogger.load(f.name)
        self.assertIn('image_vec', loaded.entries[0][1])
        self.assertIn('audio_vec', loaded.entries[0][1])


if __name__ == '__main__':
    unittest.main()
