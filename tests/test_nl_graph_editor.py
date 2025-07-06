import unittest
import importlib.machinery
import importlib.util
import types
import sys

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg

loader = importlib.machinery.SourceFileLoader('asi.graph_of_thought', 'src/graph_of_thought.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
mod = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = mod
loader.exec_module(mod)
GraphOfThought = mod.GraphOfThought

editor_loader = importlib.machinery.SourceFileLoader('asi.nl_graph_editor', 'src/nl_graph_editor.py')
editor_spec = importlib.util.spec_from_loader(editor_loader.name, editor_loader)
editor_mod = importlib.util.module_from_spec(editor_spec)
sys.modules[editor_loader.name] = editor_mod
editor_loader.exec_module(editor_mod)
NLGraphEditor = editor_mod.NLGraphEditor


class TestNLGraphEditor(unittest.TestCase):
    def test_commands(self):
        g = GraphOfThought()
        e = NLGraphEditor(g)
        e.apply('add node A')
        e.apply('add node B')
        e.apply('add edge from A to B')
        self.assertIn(1, g.edges.get(0, []))
        e.apply('merge nodes A and B')
        self.assertEqual(len(g.nodes), 1)
        nid = next(iter(g.nodes))
        self.assertIn('A', g.nodes[nid].text)
        self.assertIn('B', g.nodes[nid].text)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
