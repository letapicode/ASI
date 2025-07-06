import unittest
import importlib.machinery
import importlib.util
import sys
import types
import numpy as np

# load MultiModalWorldModel
if 'asi' not in sys.modules:
    pkg = types.ModuleType('asi')
    sys.modules['asi'] = pkg
if 'asi.loihi_backend' not in sys.modules:
    loader_lb = importlib.machinery.SourceFileLoader('asi.loihi_backend', 'src/loihi_backend.py')
    spec_lb = importlib.util.spec_from_loader(loader_lb.name, loader_lb)
    lb = importlib.util.module_from_spec(spec_lb)
    lb.__package__ = 'asi'
    sys.modules['asi.loihi_backend'] = lb
    loader_lb.exec_module(lb)
if 'asi.spiking_layers' not in sys.modules:
    loader_sl = importlib.machinery.SourceFileLoader('asi.spiking_layers', 'src/spiking_layers.py')
    spec_sl = importlib.util.spec_from_loader(loader_sl.name, loader_sl)
    sl = importlib.util.module_from_spec(spec_sl)
    sl.__package__ = 'asi'
    sys.modules['asi.spiking_layers'] = sl
    loader_sl.exec_module(sl)

if 'asi.multimodal_world_model' in sys.modules:
    mm = sys.modules['asi.multimodal_world_model']
else:
    loader_mm = importlib.machinery.SourceFileLoader('asi.multimodal_world_model', 'src/multimodal_world_model.py')
    spec_mm = importlib.util.spec_from_loader(loader_mm.name, loader_mm)
    mm = importlib.util.module_from_spec(spec_mm)
    mm.__package__ = 'asi'
    sys.modules['asi.multimodal_world_model'] = mm
    loader_mm.exec_module(mm)
MultiModalWorldModel = mm.MultiModalWorldModel
MultiModalWorldModelConfig = mm.MultiModalWorldModelConfig

# load auto_labeler
if 'asi.auto_labeler' in sys.modules:
    al = sys.modules['asi.auto_labeler']
else:
    loader_al = importlib.machinery.SourceFileLoader('asi.auto_labeler', 'src/auto_labeler.py')
    spec_al = importlib.util.spec_from_loader(loader_al.name, loader_al)
    al = importlib.util.module_from_spec(spec_al)
    al.__package__ = 'asi'
    sys.modules['asi.auto_labeler'] = al
    loader_al.exec_module(al)
AutoLabeler = al.AutoLabeler
RLLabelingAgent = al.RLLabelingAgent
RLAutoLabeler = al.RLAutoLabeler


class TestRLAutoLabeler(unittest.TestCase):
    def test_agent_update_and_select(self):
        agent = RLLabelingAgent(3, epsilon=0.0, alpha=1.0)
        agent.train([(0, 1, 1.0), (1, 2, 0.0)])
        self.assertGreater(agent.q[(0, 1)], agent.q.get((0, 0), 0.0))
        self.assertGreater(agent.performance(), 0.0)

    def test_label_with_feedback(self):
        cfg = MultiModalWorldModelConfig(vocab_size=5, img_channels=1, action_dim=2, embed_dim=8)
        model = MultiModalWorldModel(cfg)
        tok = lambda s: [ord(c) % cfg.vocab_size for c in s]
        rl_labeler = RLAutoLabeler(model, tok, agent=RLLabelingAgent(cfg.vocab_size, epsilon=0.0, alpha=1.0))
        text = "hi"
        img = np.zeros((1, 8, 8), dtype=np.float32)
        labels = rl_labeler.label_with_feedback([(text, img, None)], [0.1], [1.0])
        self.assertEqual(len(labels), 1)
        self.assertIn(labels[0], range(cfg.vocab_size))
        self.assertGreater(rl_labeler.agent.performance(), 0.0)


if __name__ == '__main__':
    unittest.main()
