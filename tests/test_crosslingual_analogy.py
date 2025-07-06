import importlib.machinery
import importlib.util
import types
import sys
import unittest

pkg = types.ModuleType('asi')
pkg.__path__ = ['src']
pkg.__spec__ = importlib.machinery.ModuleSpec('asi', None, is_package=True)
sys.modules['asi'] = pkg

# stub cryptography dependency
aes = type('AESGCM', (), {
    '__init__': lambda self, key: None,
    'encrypt': lambda self, n, d, a: d,
    'decrypt': lambda self, n, d, a: d,
})
crypto = types.ModuleType('cryptography')
haz = types.ModuleType('cryptography.hazmat')
prims = types.ModuleType('cryptography.hazmat.primitives')
ciphers = types.ModuleType('cryptography.hazmat.primitives.ciphers')
aead = types.ModuleType('cryptography.hazmat.primitives.ciphers.aead')
aead.AESGCM = aes
ciphers.aead = aead
prims.ciphers = ciphers
haz.primitives = prims
crypto.hazmat = haz
sys.modules['cryptography'] = crypto
sys.modules['cryptography.hazmat'] = haz
sys.modules['cryptography.hazmat.primitives'] = prims
sys.modules['cryptography.hazmat.primitives.ciphers'] = ciphers
sys.modules['cryptography.hazmat.primitives.ciphers.aead'] = aead


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
hm = load('asi.hierarchical_memory', 'src/hierarchical_memory.py')
clm = load('asi.cross_lingual_memory', 'src/cross_lingual_memory.py')
ca = load('asi.crosslingual_analogy_eval', 'src/crosslingual_analogy_eval.py')


class TestCrosslingualAnalogyEval(unittest.TestCase):
    def test_accuracy(self):
        path = 'tests/data/multilingual_analogies.jsonl'
        acc = ca.analogy_accuracy(path, ['es'])
        self.assertGreaterEqual(acc, 0.5)


if __name__ == '__main__':
    unittest.main()
