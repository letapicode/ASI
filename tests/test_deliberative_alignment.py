import unittest
import importlib.machinery
import importlib.util
import types
import sys

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg


def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'asi'
    loader.exec_module(mod)
    sys.modules[name] = mod
    setattr(pkg, name.split('.')[-1], mod)
    return mod

DeliberativeAligner = _load('asi.deliberative_alignment', 'src/deliberative_alignment.py').DeliberativeAligner


class TestDeliberativeAligner(unittest.TestCase):
    def test_check_and_analyze(self):
        policy = "no hacking\nno violence"
        aligner = DeliberativeAligner(policy)
        self.assertTrue(aligner.check(["say hello", "all good"]))
        self.assertFalse(aligner.check(["begin hacking sequence", "all good"]))
        text = "first step\ncommit violence"
        self.assertFalse(aligner.analyze(text))

    def test_normative_rules(self):
        aligner = DeliberativeAligner("", normative_rules=["no theft"])
        self.assertFalse(aligner.check(["plan theft"]))

    def test_check_report(self):
        aligner = DeliberativeAligner("no hacking", normative_rules=["no theft"])
        ok, flagged = aligner.check_report(["plan theft", "hack time"])
        self.assertFalse(ok)
        self.assertEqual(flagged, ["plan theft"])


if __name__ == '__main__':
    unittest.main()
