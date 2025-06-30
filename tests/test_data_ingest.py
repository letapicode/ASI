import os
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_ingest import align_triples, random_crop, generate_transcript
from PIL import Image
import wave


class TestDataIngest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        (base / "text").mkdir()
        (base / "images").mkdir()
        (base / "audio").mkdir()
        # create two samples
        for i in range(2):
            (base / "text" / f"{i}.txt").write_text(f"sample {i}")
            img = Image.new("RGB", (64, 64), color=(i * 10, 0, 0))
            img.save(base / "images" / f"{i}.png")
            with wave.open(str(base / "audio" / f"{i}.wav"), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(b"\x00\x00" * 16000)
        self.base = base

    def tearDown(self):
        self.tmp.cleanup()

    def test_align_and_aug(self):
        triples = align_triples(self.base / "text", self.base / "images", self.base / "audio")
        self.assertEqual(len(triples), 2)
        _, img_path, aud_path = triples[0]
        img = Image.open(img_path)
        crop = random_crop(img, (32, 32))
        self.assertEqual(crop.size, (32, 32))
        txt = generate_transcript(aud_path)
        self.assertIn("duration:", txt)


if __name__ == "__main__":
    unittest.main()
