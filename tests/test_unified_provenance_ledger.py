import importlib.machinery
import importlib.util
import json
import sys
import tempfile
import types
import unittest

src_pkg = types.ModuleType('src')
sys.modules['src'] = src_pkg
src_pkg.__path__ = ['src']
src_pkg.__spec__ = importlib.machinery.ModuleSpec('src', None, is_package=True)
pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg
pkg.__path__ = ['src']


def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'src'
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod

mod = _load('asi.unified_provenance_ledger', 'src/unified_provenance_ledger.py')
UnifiedProvenanceLedger = mod.UnifiedProvenanceLedger


class TestUnifiedProvenanceLedger(unittest.TestCase):
    def test_append_verify(self):
        with tempfile.TemporaryDirectory() as root:
            ledger = UnifiedProvenanceLedger(root)
            items = [
                ('dataset', json.dumps({'id': 1})),
                ('model', 'ckpt-a'),
                ('reasoning', 'step'),
            ]
            for t, r in items:
                ledger.append(t, r)
            self.assertTrue(ledger.verify(items))
            tampered = list(items)
            tampered[1] = ('model', 'ckpt-b')
            self.assertFalse(ledger.verify(tampered))


if __name__ == '__main__':
    unittest.main()
