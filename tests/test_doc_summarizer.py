import importlib.machinery
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

loader = importlib.machinery.SourceFileLoader('asi.doc_summarizer', 'src/doc_summarizer.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
doc_mod = importlib.util.module_from_spec(spec)
sys.modules['asi.doc_summarizer'] = doc_mod
loader.exec_module(doc_mod)
summarize_module = doc_mod.summarize_module


class TestDocSummarizer(unittest.TestCase):
    def test_summarize_module(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mod = Path(tmpdir) / 'dummy_mod.py'
            mod.write_text(
                """""Dummy module."""\n\n" \
                "class Foo:\n    \"\"\"Foo class.\"\"\"\n    pass\n\n" \
                "def bar():\n    \"\"\"Return 1.\"\"\"\n    return 1\n"
            )
            md = summarize_module('dummy_mod', search_path=[tmpdir])
            self.assertIn('# dummy_mod', md)
            self.assertIn('## Foo', md)
            self.assertIn('## bar', md)

    def test_cli_writes_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mod = Path(tmpdir) / 'dummy_cli.py'
            mod.write_text(
                """""Dummy CLI."""\n\n" \
                "def func():\n    \"\"\"do nothing\"\"\"\n    pass\n"
            )
            out_dir = Path(tmpdir) / 'docs'
            env = os.environ.copy()
            env['PYTHONPATH'] = f"{Path('.').resolve()}:{tmpdir}"
            proc = subprocess.run(
                [sys.executable, '-m', 'asi.doc_summarizer', 'dummy_cli', '--out-dir', str(out_dir)],
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            out_file = out_dir / 'dummy_cli.md'
            self.assertTrue(out_file.exists())


if __name__ == '__main__':
    unittest.main()
