import json
import tempfile
import unittest
from pathlib import Path
from asi.license_inspector import LicenseInspector


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
