import unittest
import importlib.machinery
import importlib.util
import types
import sys

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg


def load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'asi'
    loader.exec_module(mod)
    sys.modules[name] = mod
    setattr(pkg, name.split('.')[-1], mod)
    return mod


fe = load('asi.fairness', 'src/fairness.py')

class CrossLingualTranslator:
    def __init__(self, languages):
        self.languages = list(languages)

    def translate(self, text, lang):
        if lang not in self.languages:
            raise ValueError('unsupported language')
        return f'[{lang}] {text}'

    def translate_all(self, text):
        return {l: self.translate(text, l) for l in self.languages}


dummy = types.ModuleType('asi.data_ingest')
dummy.CrossLingualTranslator = CrossLingualTranslator
sys.modules['asi.data_ingest'] = dummy

clf = load('asi.cross_lingual_fairness', 'src/fairness.py')

CrossLingualFairnessEvaluator = clf.CrossLingualFairnessEvaluator


class TestCrossLingualFairness(unittest.TestCase):
    def test_group_aggregation(self):
        tr = CrossLingualTranslator(['en'])
        ev = CrossLingualFairnessEvaluator(translator=tr)
        stats = {
            'hola': {'tp': 1, 'fn': 1},
            '[en] hola': {'tp': 2, 'fn': 0},
        }
        res = ev.evaluate(stats, positive_label='tp')
        self.assertGreater(res['demographic_parity'], 0.0)
        self.assertGreater(res['equal_opportunity'], 0.0)


if __name__ == '__main__':
    unittest.main()
