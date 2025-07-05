import unittest
import importlib.machinery
import importlib.util
import sys

loader = importlib.machinery.SourceFileLoader('got_visualizer', 'src/got_visualizer.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
mod = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = mod
loader.exec_module(mod)
GOTVisualizer = mod.GOTVisualizer


class TestGOTVisualizer(unittest.TestCase):
    def test_html_generation(self):
        nodes = [{"id": "a:0", "text": "start"}, {"id": "a:1", "text": "end"}]
        edges = [("a:0", "a:1")]
        vis = GOTVisualizer(nodes, edges)
        html = vis.to_html()
        self.assertIn("<html", html.lower())


if __name__ == "__main__":
    unittest.main()
