import importlib.machinery
import importlib.util
import tempfile
import unittest
from pathlib import Path

loader = importlib.machinery.SourceFileLoader('asi.dataset_summarizer', 'src/dataset_summarizer.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
mod = importlib.util.module_from_spec(spec)
loader.exec_module(mod)
summarize_dataset = mod.summarize_dataset


class TestDatasetSummarizer(unittest.TestCase):
    def test_summarize_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'a.txt').write_text('foo bar baz')
            (root / 'b.txt').write_text('foo qux qux')
            summaries = summarize_dataset(root)
            self.assertTrue(len(summaries) > 0)


if __name__ == '__main__':
    unittest.main()
