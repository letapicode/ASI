import unittest
import importlib.machinery
import importlib.util
import types
import sys

sys.modules['torch'] = types.ModuleType('torch')
sys.modules['requests'] = types.ModuleType('requests')
sys.modules['aiohttp'] = types.ModuleType('aiohttp')
sys.modules['PIL'] = types.ModuleType('PIL')
sys.modules['PIL.Image'] = types.ModuleType('PIL.Image')

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


di = load('asi.data_ingest', 'src/data_ingest.py')
at = load('asi.adaptive_translator', 'src/adaptive_translator.py')

CrossLingualTranslator = di.CrossLingualTranslator
AdaptiveTranslator = at.AdaptiveTranslator


class TestAdaptiveTranslator(unittest.TestCase):
    def test_reward_guides_translation(self):
        tr = CrossLingualTranslator(['en', 'fr'])
        adapt = AdaptiveTranslator(tr, epsilon=0.0, alpha=1.0)
        tr.adaptive = adapt
        first = tr.translate('hello')
        self.assertEqual(first, '[en] hello')
        adapt.update(-1.0, lang='en')
        adapt.update(1.0, lang='fr')
        second = tr.translate('world')
        self.assertEqual(second, '[fr] world')


if __name__ == '__main__':
    unittest.main()
