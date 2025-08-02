import json
import tempfile
import unittest
from pathlib import Path
import importlib.machinery
import importlib.util
import types
import sys

sys.modules['scripts'] = types.ModuleType('scripts')


def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = name.rpartition('.')[0]
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod

# prepare src package before loading script
src_pkg = types.ModuleType('src')
sys.modules['src'] = src_pkg
src_pkg.__path__ = ['src']
src_pkg.__spec__ = importlib.machinery.ModuleSpec('src', None, is_package=True)

DatasetLineageManager = _load('src.dataset_lineage', 'src/dataset_lineage.py').DatasetLineageManager

summary_mod = _load('scripts.dataset_summary', 'scripts/dataset_summary.py')
summarize = summary_mod.summarize


class TestDatasetSummary(unittest.TestCase):
    def test_summarize_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inp = root / 'in.txt'
            outp = root / 'out.txt'
            inp.write_text('i')
            outp.write_text('o')
            mgr = DatasetLineageManager(root)
            mgr.record([inp], [outp], note='step1')
            meta = root / 'meta.json'
            meta.write_text(json.dumps({'license': 'MIT'}))
            result = summarize(str(root), fmt='json')
            data = json.loads(result)
            self.assertEqual(data['lineage'][0]['note'], 'step1')
            self.assertTrue(data['licenses'][str(meta)])

    def test_summarize_with_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inp = root / 's.txt'
            inp.write_text('hello world world')
            mgr = DatasetLineageManager(root)
            mgr.record([inp], [inp], note='c')
            meta = root / 'm.json'
            meta.write_text(json.dumps({'license': 'MIT'}))
            result = summarize(str(root), fmt='json', content=True)
            data = json.loads(result)
            self.assertIn('content_summaries', data)

    def test_fairness_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stats = {'g1': {'tp': 1, 'fn': 1}, 'g2': {'tp': 2, 'fn': 0}}
            stats_file = root / 'fairness.json'
            stats_file.write_text(json.dumps(stats))
            out_dir = Path('docs/datasets')
            if out_dir.exists():
                for p in out_dir.iterdir():
                    p.unlink()
            summary_mod.main([str(root), '--fairness-report', str(stats_file)])
            out_img = out_dir / f'{root.name}_fairness.png'
            self.assertTrue(out_img.exists())


if __name__ == '__main__':
    unittest.main()
