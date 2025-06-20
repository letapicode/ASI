import unittest
import tempfile
from pathlib import Path

from src.autobench import run_autobench


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


if __name__ == "__main__":
    unittest.main()
