import tempfile
import json
import unittest
import importlib.machinery
import importlib.util
import sys
import types

src_pkg = types.ModuleType('src')
sys.modules['src'] = src_pkg
src_pkg.__path__ = ['src']
src_pkg.__spec__ = importlib.machinery.ModuleSpec('src', None, is_package=True)
loader = importlib.machinery.SourceFileLoader('src.provenance_ledger', 'src/provenance_ledger.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
mod = importlib.util.module_from_spec(spec)
mod.__package__ = 'src'
sys.modules['src.provenance_ledger'] = mod
sys.modules['asi.provenance_ledger'] = mod
loader.exec_module(mod)
BlockchainProvenanceLedger = mod.BlockchainProvenanceLedger


class TestBlockchainProvenance(unittest.TestCase):
    def test_append_and_verify(self):
        with tempfile.TemporaryDirectory() as root:
            ledger = BlockchainProvenanceLedger(root)
            recs = [json.dumps({'i': i}) for i in range(3)]
            for r in recs:
                ledger.append(r)
            self.assertTrue(ledger.verify(recs))
            # tamper with a record
            recs[1] = json.dumps({'i': 99})
            self.assertFalse(ledger.verify(recs))


if __name__ == '__main__':
    unittest.main()
