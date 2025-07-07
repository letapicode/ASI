import unittest
import importlib.machinery
import importlib.util
import runpy
import sys
from pathlib import Path

loader = importlib.machinery.SourceFileLoader('asi.code_refine', 'src/code_refine.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
mod = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = mod
loader.exec_module(mod)
CodeRefinePipeline = mod.CodeRefinePipeline


class TestCodeRefinePipeline(unittest.TestCase):
    def test_refine_basic(self):
        src = 'def foo(a,b):\n    if a == None:\n        return b\n'
        refined = CodeRefinePipeline().refine(src)
        self.assertIn('from __future__ import annotations', refined)
        self.assertIn('from typing import Any', refined)
        self.assertIn('a: Any', refined)
        self.assertIn('b: Any', refined)
        self.assertIn('a is None', refined)
        self.assertNotIn('== None', refined)

    def test_refine_extended(self):
        src = (
            'async def bar(*args, **kwargs):\n'
            '    if None == kwargs.get("x") or kwargs.get("y") == True:\n'
            '        return args\n'
        )
        refined = CodeRefinePipeline().refine(src)
        self.assertIn('*args: Any', refined)
        self.assertIn('**kwargs: Any', refined)
        self.assertIn("kwargs.get('x') is None", refined)
        self.assertIn("kwargs.get('y') is True", refined)

    def test_cli_dry_run_and_write(self):
        tmp = Path(self.tmpdir)
        f = tmp / 'temp.py'
        f.write_text('def foo():\n    return True == False\n')

        self._run_cli([str(f), '--dry-run'])
        self.assertIn('True == False', f.read_text())

        self._run_cli([str(f)])
        self.assertIn('True is False', f.read_text())

    def _run_cli(self, argv):
        saved = sys.argv
        sys.argv = ['code_refine.py'] + argv
        try:
            runpy.run_path('scripts/code_refine.py', run_name='__main__')
        finally:
            sys.argv = saved

    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)




if __name__ == '__main__':
    unittest.main()
