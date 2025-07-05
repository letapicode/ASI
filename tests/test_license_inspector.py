import json
import tempfile
import unittest
from pathlib import Path
import importlib.machinery
import importlib.util
import types
import sys

src_pkg = types.ModuleType('src')
src_pkg.__path__ = ['src']
src_pkg.__spec__ = importlib.machinery.ModuleSpec('src', None, is_package=True)
sys.modules['src'] = src_pkg
loader = importlib.machinery.SourceFileLoader('src.license_inspector', 'src/license_inspector.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
li = importlib.util.module_from_spec(spec)
li.__package__ = 'src'
sys.modules['src.license_inspector'] = li
loader.exec_module(li)
LicenseInspector = li.LicenseInspector


class TestLicenseInspector(unittest.TestCase):
    def test_inspect(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = Path(tmp) / 'sample.json'
            meta.write_text(json.dumps({'license': 'MIT'}))
            insp = LicenseInspector()
            res = insp.inspect_dir(tmp)
            self.assertTrue(res[str(meta)])


if __name__ == '__main__':
    unittest.main()
