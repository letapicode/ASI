import json
import os
import subprocess
import sys
import tempfile
import unittest
import types
import json

from asi.reasoning_history import ReasoningHistoryLogger


class CrossLingualTranslator:
    def __init__(self, languages):
        self.languages = list(languages)

    def translate(self, text, lang):
        if lang not in self.languages:
            raise ValueError('unsupported language')
        return f'[{lang}] {text}'

    def translate_all(self, text):
        return {l: self.translate(text, l) for l in self.languages}


dummy = types.ModuleType('asi.data_ingest')
dummy.CrossLingualTranslator = CrossLingualTranslator
sys.modules['asi.data_ingest'] = dummy


class TestHistoryTranslation(unittest.TestCase):
    def test_logger_translates(self):
        tr = CrossLingualTranslator(["es"])
        logger = ReasoningHistoryLogger(translator=tr)
        logger.log("start -> end")
        entries = logger.get_history()
        self.assertIsInstance(entries[0][1], dict)
        self.assertEqual(entries[0][1]["translations"]["es"], "[es] start -> end")


class TestSelfReflectionCLITranslations(unittest.TestCase):
    def test_cli_with_translations(self):
        tr = CrossLingualTranslator(["es"])
        logger = ReasoningHistoryLogger(translator=tr)
        logger.log("start -> not start")
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
