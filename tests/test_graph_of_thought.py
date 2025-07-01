import json
import subprocess
import sys
import tempfile
import unittest
import os
import importlib.machinery
import importlib.util

loader = importlib.machinery.SourceFileLoader(
    'graph_of_thought', 'src/graph_of_thought.py'
)
spec = importlib.util.spec_from_loader(loader.name, loader)
graph_mod = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = graph_mod
loader.exec_module(graph_mod)
GraphOfThought = graph_mod.GraphOfThought


class TestGraphOfThought(unittest.TestCase):
    def test_add_and_search(self):
        g = GraphOfThought()
        a = g.add_step("start")
        b = g.add_step("analyze")
        c = g.add_step("apply refactor")
        g.connect(a, b)
        g.connect(b, c)
        path = g.plan_refactor(a, keyword="refactor")
        self.assertEqual(path, [a, b, c])

    def test_unreachable(self):
        g = GraphOfThought()
        a = g.add_step("start")
        b = g.add_step("middle")
        g.connect(a, b)
        path = g.plan_refactor(a, keyword="refactor")
        self.assertEqual(path, [])


class TestGraphOfThoughtCLI(unittest.TestCase):
    def test_cli_runs(self):
        data = {
            "nodes": [
                {"id": 0, "text": "start"},
                {"id": 1, "text": "apply refactor"},
            ],
            "edges": [[0, 1]],
        }
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            json.dump(data, f)
            fname = f.name
        try:
            proc = subprocess.run(
                [sys.executable, "src/graph_of_thought.py", fname, "0"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0)
            self.assertIn("0 -> 1", proc.stdout)
        finally:
            os.unlink(fname)


if __name__ == "__main__":  # pragma: no cover - test helper
    unittest.main()
