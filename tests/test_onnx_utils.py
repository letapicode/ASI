import unittest
import tempfile
import importlib.machinery
import importlib.util
import sys
import numpy as np
import onnxruntime as ort

# preload dependencies for relative imports
loader = importlib.machinery.SourceFileLoader('asi.lora_quant', 'src/lora_quant.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
mod_lora = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = mod_lora
loader.exec_module(mod_lora)

loader = importlib.machinery.SourceFileLoader('asi.multimodal_world_model', 'src/multimodal_world_model.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
mmwm = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = mmwm
loader.exec_module(mmwm)
MultiModalWorldModelConfig = mmwm.MultiModalWorldModelConfig
MultiModalWorldModel = mmwm.MultiModalWorldModel

loader = importlib.machinery.SourceFileLoader('asi.cross_modal_fusion', 'src/cross_modal_fusion.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
cmf = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = cmf
loader.exec_module(cmf)
CrossModalFusionConfig = cmf.CrossModalFusionConfig
CrossModalFusion = cmf.CrossModalFusion

loader = importlib.machinery.SourceFileLoader('asi.onnx_utils', 'src/onnx_utils.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
onnx_utils = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = onnx_utils
loader.exec_module(onnx_utils)
export_to_onnx = onnx_utils.export_to_onnx


class TestONNXUtils(unittest.TestCase):
    def test_world_model_export(self):
        cfg = MultiModalWorldModelConfig(vocab_size=10, img_channels=3, action_dim=2, embed_dim=8)
        model = MultiModalWorldModel(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = tmpdir + '/wm.onnx'
            export_to_onnx(model, path)
            sess = ort.InferenceSession(path)
            inputs = {
                'text': np.random.randint(0, cfg.vocab_size, (1, 4), dtype=np.int64),
                'image': np.random.randn(1, cfg.img_channels, 8, 8).astype(np.float32),
                'action': np.random.randint(0, cfg.action_dim, (1,), dtype=np.int64),
            }
            out = sess.run(None, inputs)
            self.assertEqual(len(out), 2)

    def test_fusion_export(self):
        cfg = CrossModalFusionConfig(vocab_size=10, text_dim=8, img_channels=3, audio_channels=1, latent_dim=4)
        model = CrossModalFusion(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = tmpdir + '/fusion.onnx'
            export_to_onnx(model, path)
            sess = ort.InferenceSession(path)
            inputs = {
                'text': np.random.randint(0, cfg.vocab_size, (1, 5), dtype=np.int64),
                'images': np.random.randn(1, cfg.img_channels, 16, 16).astype(np.float32),
                'audio': np.random.randn(1, cfg.audio_channels, 32).astype(np.float32),
            }
            out = sess.run(None, inputs)
            self.assertEqual(len(out), 3)


if __name__ == '__main__':
    unittest.main()
