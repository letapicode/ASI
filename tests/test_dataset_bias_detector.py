import unittest
import tempfile
from pathlib import Path
import importlib.machinery
import importlib.util
import types
import sys

pkg = types.ModuleType('asi')
pkg.__path__ = ['src']
sys.modules['asi'] = pkg

loader_tel = importlib.machinery.SourceFileLoader('asi.telemetry', 'src/telemetry.py')
spec_tel = importlib.util.spec_from_loader(loader_tel.name, loader_tel)
tel_mod = importlib.util.module_from_spec(spec_tel)
sys.modules['asi.telemetry'] = tel_mod
loader_tel.exec_module(tel_mod)

loader_bias = importlib.machinery.SourceFileLoader('asi.dataset_bias_detector', 'src/dataset_bias_detector.py')
spec_bias = importlib.util.spec_from_loader(loader_bias.name, loader_bias)
bias_mod = importlib.util.module_from_spec(spec_bias)
sys.modules['asi.dataset_bias_detector'] = bias_mod
loader_bias.exec_module(bias_mod)

loader_filt = importlib.machinery.SourceFileLoader('asi.auto_dataset_filter', 'src/auto_dataset_filter.py')
spec_filt = importlib.util.spec_from_loader(loader_filt.name, loader_filt)
filt_mod = importlib.util.module_from_spec(spec_filt)
sys.modules['asi.auto_dataset_filter'] = filt_mod
loader_filt.exec_module(filt_mod)

compute_word_freq = bias_mod.compute_word_freq
bias_score = bias_mod.bias_score
text_bias_score = bias_mod.text_bias_score
DatasetBiasDetector = bias_mod.DatasetBiasDetector
filter_text_files = filt_mod.filter_text_files


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

    def test_stream_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.txt"
            p.write_text("hello world")
            det = DatasetBiasDetector()
            det.score_file(p)
            metrics = list(det.stream_metrics())
            self.assertEqual(len(metrics), 1)
            self.assertIn("bias_score", metrics[0])


if __name__ == "__main__":
    unittest.main()
