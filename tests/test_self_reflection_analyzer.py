import json
import os
import subprocess
import sys
import tempfile
import unittest

from asi.reasoning_history import ReasoningHistoryLogger


class TestHistoryAnalysis(unittest.TestCase):
    def test_analyze(self):
        logger = ReasoningHistoryLogger()
        logger.log("start -> middle")
        logger.log("start -> not middle")
        analysis = logger.analyze()
        self.assertEqual(analysis.clusters.get("start"), 2)
        self.assertIn(("middle", "not middle"), analysis.inconsistencies)


class TestSelfReflectionCLI(unittest.TestCase):
    def test_cli_report(self):
        logger = ReasoningHistoryLogger()
        logger.log("start -> middle")
        logger.log("start -> not middle")
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            json.dump(logger.entries, f)
            fname = f.name
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "asi.self_reflection", fname],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0)
            self.assertIn("Inconsistencies", proc.stdout)
        finally:
            os.unlink(fname)


if __name__ == "__main__":  # pragma: no cover - test helper
    unittest.main()
