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

NormativeReasoner = _load('asi.normative_reasoner', 'src/normative_reasoner.py').NormativeReasoner


def test_normative_reasoner_check():
    nr = NormativeReasoner(["no theft", "no harm"])
    steps = ["say hello", "plan theft"]
    ok, flagged = nr.check(steps)
    assert not ok
    assert flagged == ["plan theft"]


def test_normative_reasoner_regex():
    nr = NormativeReasoner([r"plan\s+theft"], use_regex=True)
    ok, flagged = nr.check(["ok", "plan    theft"])
    assert not ok
    assert flagged == ["plan    theft"]


def test_normative_reasoner_fuzzy():
    nr = NormativeReasoner(["theft"], fuzzy_threshold=0.8)
    ok, flagged = nr.check(["thieft"])  # misspelled
    assert not ok
    assert flagged == ["thieft"]
