import os
import sys
from pathlib import Path
import tempfile
import random
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_ingest import (
    pair_modalities,
    random_crop_image,
    add_gaussian_noise,
    text_dropout,
)


class TestDataIngest(unittest.TestCase):
    def test_pair_modalities(self):
        with tempfile.TemporaryDirectory() as root:
            td = os.path.join(root, "text")
            id = os.path.join(root, "img")
            ad = os.path.join(root, "aud")
            os.mkdir(td)
            os.mkdir(id)
            os.mkdir(ad)
            open(os.path.join(td, "sample.txt"), "w").close()
            open(os.path.join(id, "sample.jpg"), "w").close()
            open(os.path.join(ad, "sample.wav"), "w").close()
            pairs = pair_modalities(td, id, ad)
            self.assertEqual(
                pairs,
                [
                    (
                        os.path.join(td, "sample.txt"),
                        os.path.join(id, "sample.jpg"),
                        os.path.join(ad, "sample.wav"),
                    )
                ],
            )

    def test_augmentations(self):
        random.seed(0)
        img = np.arange(16).reshape(4, 4, 1)
        crop = random_crop_image(img, (2, 2))
        self.assertEqual(crop.shape, (2, 2, 1))

        audio = np.zeros(10)
        noisy = add_gaussian_noise(audio, std=0.1)
        self.assertEqual(noisy.shape, audio.shape)

        text = "the quick brown fox"
        out = text_dropout(text, p=1.0)
        self.assertTrue(out)


if __name__ == "__main__":
    unittest.main()
