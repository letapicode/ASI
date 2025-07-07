import json
import tempfile
import unittest
from pathlib import Path
import importlib.machinery
import importlib.util
import types
import sys

src_pkg = types.ModuleType('src')
sys.modules['src'] = src_pkg
src_pkg.__path__ = ['src']
src_pkg.__spec__ = importlib.machinery.ModuleSpec('src', None, is_package=True)

loader_pbm = importlib.machinery.SourceFileLoader('src.privacy_budget_manager', 'src/privacy_budget_manager.py')
spec_pbm = importlib.util.spec_from_loader(loader_pbm.name, loader_pbm)
pbm = importlib.util.module_from_spec(spec_pbm)
pbm.__package__ = 'src'
sys.modules['src.privacy_budget_manager'] = pbm
loader_pbm.exec_module(pbm)

loader_li = importlib.machinery.SourceFileLoader('src.license_inspector', 'src/license_inspector.py')
spec_li = importlib.util.spec_from_loader(loader_li.name, loader_li)
li = importlib.util.module_from_spec(spec_li)
li.__package__ = 'src'
sys.modules['src.license_inspector'] = li
loader_li.exec_module(li)

loader_dlm = importlib.machinery.SourceFileLoader('src.dataset_lineage_manager', 'src/dataset_lineage_manager.py')
spec_dlm = importlib.util.spec_from_loader(loader_dlm.name, loader_dlm)
dlm = importlib.util.module_from_spec(spec_dlm)
dlm.__package__ = 'src'
sys.modules['src.dataset_lineage_manager'] = dlm
loader_dlm.exec_module(dlm)

loader_pa = importlib.machinery.SourceFileLoader('src.privacy_auditor', 'src/privacy_auditor.py')
spec_pa = importlib.util.spec_from_loader(loader_pa.name, loader_pa)
pa = importlib.util.module_from_spec(spec_pa)
pa.__package__ = 'src'
sys.modules['src.privacy_auditor'] = pa
loader_pa.exec_module(pa)

PrivacyBudgetManager = pbm.PrivacyBudgetManager
LicenseInspector = li.LicenseInspector
DatasetLineageManager = dlm.DatasetLineageManager
PrivacyAuditor = pa.PrivacyAuditor


class TestPrivacyAuditor(unittest.TestCase):
    def test_audit_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            pbm_mgr = PrivacyBudgetManager(1.0, 1e-5, Path(tmp)/'b.json')
            insp = LicenseInspector(['mit'])
            lineage = DatasetLineageManager(tmp)
            auditor = PrivacyAuditor(pbm_mgr, insp, lineage, report_dir=tmp)
            t = Path(tmp)/'t.txt'
            t.write_text('x')
            meta = t.with_suffix('.json')
            meta.write_text(json.dumps({'license': 'MIT'}))
            i = Path(tmp)/'i.png'
            i.write_bytes(b'')
            a = Path(tmp)/'a.wav'
            a.write_bytes(b'')
            auditor.audit_triple((t, i, a), meta, run_id='r')
            rpt = auditor.write_report('r')
            self.assertTrue(Path(rpt).exists())
            data = json.loads(Path(rpt).read_text())
            self.assertIn('remaining_epsilon', data)


if __name__ == '__main__':
    unittest.main()
