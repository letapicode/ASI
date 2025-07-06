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

logger_loader = importlib.machinery.SourceFileLoader(
    'reasoning_history', 'src/reasoning_history.py'
)
logger_spec = importlib.util.spec_from_loader(logger_loader.name, logger_loader)
logger_mod = importlib.util.module_from_spec(logger_spec)
sys.modules[logger_loader.name] = logger_mod
logger_loader.exec_module(logger_mod)
ReasoningHistoryLogger = logger_mod.ReasoningHistoryLogger

mem_loader = importlib.machinery.SourceFileLoader(
    'context_summary_memory', 'src/context_summary_memory.py'
)
mem_spec = importlib.util.spec_from_loader(mem_loader.name, mem_loader)
mem_mod = importlib.util.module_from_spec(mem_spec)
sys.modules[mem_loader.name] = mem_mod
mem_loader.exec_module(mem_mod)
ContextSummaryMemory = mem_mod.ContextSummaryMemory


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

    def test_self_reflect_and_logger(self):
        g = GraphOfThought()
        a = g.add_step("start")
        b = g.add_step("middle")
        c = g.add_step("end")
        g.connect(a, b)
        g.connect(b, c)
        summary = g.self_reflect()
        self.assertEqual(summary, "start -> middle -> end")

        logger = ReasoningHistoryLogger()
        logger.log(summary)
        hist = logger.get_history()
        self.assertEqual(len(hist), 1)
        self.assertEqual(hist[0][1], summary)

    def test_plan_refactor_with_summary(self):
        g = GraphOfThought()
        a = g.add_step("start")
        b = g.add_step("analyze")
        c = g.add_step("apply refactor")
        g.connect(a, b)
        g.connect(b, c)

        class DummySummarizer:
            def summarize(self, text):
                return "sum"

            def expand(self, text):
                return 0

        mem = ContextSummaryMemory(
            dim=2, compressed_dim=1, capacity=2, summarizer=DummySummarizer()
        )
        path, summary = g.plan_refactor(
            a, summary_memory=mem, summary_threshold=1
        )
        self.assertEqual(path, [a, b, c])
        self.assertEqual(summary, "sum")


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
