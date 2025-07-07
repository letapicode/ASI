import unittest
import json
import tempfile
import json
import importlib.machinery
import importlib.util
import types
import sys

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg
sys.modules['scripts'] = types.ModuleType('scripts')
pkg.__path__ = ['src']


def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = name.rpartition('.')[0]
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod

# minimal dependencies for ab_evaluator
_load('asi.eval_harness', 'src/eval_harness.py')
_load('asi.ab_evaluator', 'src/ab_evaluator.py')
ape = _load('scripts.ab_prompt_eval', 'scripts/ab_prompt_eval.py')


class TestABPromptEval(unittest.TestCase):
    def test_output_contains_delta(self):
        with tempfile.NamedTemporaryFile('w+', suffix='.json') as a, tempfile.NamedTemporaryFile('w+', suffix='.json') as b:
            json.dump({'modules': []}, a)
            json.dump({'modules': []}, b)
            a.flush()
            b.flush()
            # capture print output
            from io import StringIO
            import contextlib
            buf = StringIO()
            with contextlib.redirect_stdout(buf):
                ape.main(['--config-a', a.name, '--config-b', b.name])
            out = buf.getvalue()
            self.assertIn('Engagement A', out)


if __name__ == '__main__':
    unittest.main()
