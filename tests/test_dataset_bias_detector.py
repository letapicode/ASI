import unittest
import tempfile
from pathlib import Path

from asi.dataset_bias_detector import compute_word_freq, bias_score, text_bias_score
from asi.auto_dataset_filter import filter_text_files


class TestDatasetBiasDetector(unittest.TestCase):
    def test_bias_score_and_filter(self):
        with tempfile.TemporaryDirectory() as d:
            p1 = Path(d) / "a.txt"
            p2 = Path(d) / "b.txt"
            p1.write_text("hello hello hello")
            p2.write_text("alpha beta gamma delta")
            freq = compute_word_freq([p1, p2])
            score = bias_score(freq)
            self.assertGreaterEqual(score, 0.0)

            kept = filter_text_files([p1, p2], bias_threshold=0.8)
            self.assertEqual(kept, [p2])


if __name__ == "__main__":
    unittest.main()
