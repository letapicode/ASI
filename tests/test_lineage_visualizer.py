import tempfile
import unittest
from pathlib import Path
import importlib.machinery
import importlib.util
import sys
import types


src_pkg = types.ModuleType('src')
sys.modules['src'] = src_pkg
src_pkg.__path__ = ['src']
src_pkg.__spec__ = importlib.machinery.ModuleSpec('src', None, is_package=True)


def _load(name: str, path: str):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod


DatasetLineageManager = _load('src.dataset_lineage', 'src/dataset_lineage.py').DatasetLineageManager
LineageVisualizer = _load('src.graph_visualizers', 'src/graph_visualizers.py').LineageVisualizer


class TestLineageVisualizer(unittest.TestCase):
    def test_graph_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inp = root / 'in.txt'
            outp = root / 'out.txt'
            inp.write_text('a')
            outp.write_text('b')
            mgr = DatasetLineageManager(root)
            mgr.record([inp], [outp], note='copy')
            viz = LineageVisualizer(mgr)
            data = viz.graph_json()
            self.assertEqual(len(data['nodes']), 2)
            self.assertEqual(len(data['links']), 1)
            link = data['links'][0]
            self.assertEqual(link['source'], str(inp))
            self.assertEqual(link['target'], str(outp))
            self.assertEqual(link['note'], 'copy')


if __name__ == '__main__':
    unittest.main()
