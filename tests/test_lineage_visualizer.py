import tempfile
import unittest
from pathlib import Path
import importlib.machinery
import importlib.util
import sys


def _load(name: str, path: str):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    sys.modules[name] = mod
    return mod


DatasetLineageManager = _load('src.dataset_lineage_manager', 'src/dataset_lineage_manager.py').DatasetLineageManager
LineageVisualizer = _load('src.lineage_visualizer', 'src/lineage_visualizer.py').LineageVisualizer


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
