import unittest
import tempfile
from pathlib import Path

import importlib.machinery
import importlib.util

loader = importlib.machinery.SourceFileLoader('autobench', 'src/autobench.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
autobench = importlib.util.module_from_spec(spec)
loader.exec_module(autobench)
BenchResult = autobench.BenchResult
run_autobench = autobench.run_autobench
summarize_results = autobench.summarize_results


class TestAutoBench(unittest.TestCase):
    def test_run_autobench(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test_dummy.py"
            test_file.write_text(
                "import unittest\n\n"
                "class Dummy(unittest.TestCase):\n"
                "    def test_ok(self):\n"
                "        self.assertEqual(2 + 2, 4)\n\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            )
            results = run_autobench(tmpdir)
            self.assertIn("test_dummy.py", results)
            self.assertTrue(results["test_dummy.py"].passed)

    def test_summarize_results(self):
        results = {
            "a.py": BenchResult(True, ""),
            "b.py": BenchResult(False, "Line1\nLine2\nLine3\nLine4"),
        }
        summary = summarize_results(results)
        self.assertIn("Passed 1/2 modules", summary)
        self.assertIn("a.py: PASS", summary)
        self.assertIn("b.py: FAIL", summary)
        self.assertIn("Line1", summary)
        self.assertIn("Line3", summary)
        self.assertNotIn("Line4", summary)


if __name__ == "__main__":
    unittest.main()
