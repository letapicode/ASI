import unittest
import numpy as np
import tempfile
import wave
from pathlib import Path
import importlib.machinery
import importlib.util
import types
import sys
from PIL import Image

src_pkg = types.ModuleType('src')
src_pkg.__path__ = ['src']
src_pkg.__spec__ = importlib.machinery.ModuleSpec('src', None, is_package=True)
sys.modules['src'] = src_pkg
loader = importlib.machinery.SourceFileLoader('src.dataset_anonymizer', 'src/dataset_anonymizer.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
mod = importlib.util.module_from_spec(spec)
mod.__package__ = 'src'
sys.modules['src.dataset_anonymizer'] = mod
loader.exec_module(mod)
DatasetAnonymizer = mod.DatasetAnonymizer


class TestDatasetAnonymizer(unittest.TestCase):
    def test_scrub_text(self):
        da = DatasetAnonymizer()
        out = da.scrub_text("contact me at foo@bar.com or 123-456-7890")
        self.assertNotIn("@", out)
        self.assertNotIn("123-456-7890", out)
        self.assertEqual(da.summary()["text"], 1)

    def test_scrub_files(self):
        da = DatasetAnonymizer()
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp) / "t.txt"
            i = Path(tmp) / "i.png"
            a = Path(tmp) / "a.wav"
            t.write_text("a@b.com")
            img = Image.new("RGB", (2, 2), color=1)
            img.save(i)
            with wave.open(str(a), "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(8000)
                f.writeframes(np.ones(8, dtype=np.int16).tobytes())
            da.scrub_text_file(t)
            da.scrub_image_file(i)
            da.scrub_audio_file(a)
            self.assertEqual(t.read_text(), "[EMAIL]")
            self.assertTrue(np.all(np.array(Image.open(i)) == 0))
            with wave.open(str(a)) as f:
                data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
            self.assertTrue(np.all(data == 0))
            summ = da.summary()
            self.assertEqual(summ["text"], 1)
            self.assertEqual(summ["image"], 1)
            self.assertEqual(summ["audio"], 1)


if __name__ == "__main__":
    unittest.main()
