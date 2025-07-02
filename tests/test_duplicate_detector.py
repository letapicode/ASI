import unittest
from asi.duplicate_detector import DuplicateDetector


class TestDuplicateDetector(unittest.TestCase):
    def test_filter_texts(self):
        det = DuplicateDetector(threshold=0.99)
        texts = ["hello world", "hello world", "hi there"]
        kept = det.filter_texts(texts)
        self.assertEqual(len(kept), 2)
        self.assertIn("hello world", kept)
        self.assertIn("hi there", kept)


if __name__ == "__main__":
    unittest.main()
