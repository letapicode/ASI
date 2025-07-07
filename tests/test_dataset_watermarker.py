import tempfile
import unittest
from pathlib import Path
import importlib.machinery
import importlib.util
import sys
import types

src_pkg = types.ModuleType('src')
sys.modules['src'] = src_pkg
src_pkg.__path__ = ['src']
src_pkg.__spec__ = importlib.machinery.ModuleSpec('src', None, is_package=True)
loader = importlib.machinery.SourceFileLoader('src.dataset_watermarker', 'src/dataset_watermarker.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
wm_mod = importlib.util.module_from_spec(spec)
wm_mod.__package__ = 'src'
sys.modules['src.dataset_watermarker'] = wm_mod
loader.exec_module(wm_mod)
add_watermark = wm_mod.add_watermark
detect_watermark = wm_mod.detect_watermark

from PIL import Image
import numpy as np
import wave


class TestDatasetWatermarker(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            txt = root / 't.txt'
            txt.write_text('hello')
            add_watermark(txt, 'id1')
            self.assertEqual(detect_watermark(txt), 'id1')

            img_path = root / 'i.png'
            Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(img_path)
            add_watermark(img_path, 'id2')
            self.assertEqual(detect_watermark(img_path), 'id2')

            aud_path = root / 'a.wav'
            with wave.open(str(aud_path), 'wb') as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(np.zeros(10, dtype=np.int16).tobytes())
            add_watermark(aud_path, 'id3')
            self.assertEqual(detect_watermark(aud_path), 'id3')


if __name__ == '__main__':
    unittest.main()
