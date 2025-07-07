import json
import os
import subprocess
import sys
import tempfile
import unittest
import types
from pathlib import Path

import importlib.machinery
import importlib.util

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg

loader_rh = importlib.machinery.SourceFileLoader(
    'asi.reasoning_history', 'src/reasoning_history.py'
)
spec_rh = importlib.util.spec_from_loader(loader_rh.name, loader_rh)
rh_mod = importlib.util.module_from_spec(spec_rh)
rh_mod.__package__ = 'asi'
sys.modules['asi.reasoning_history'] = rh_mod
loader_rh.exec_module(rh_mod)
ReasoningHistoryLogger = rh_mod.ReasoningHistoryLogger


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

loader = importlib.machinery.SourceFileLoader(
    'asi.reasoning_summary_translator', 'src/reasoning_summary_translator.py'
)
spec = importlib.util.spec_from_loader(loader.name, loader)
rst_mod = importlib.util.module_from_spec(spec)
rst_mod.__package__ = 'asi'
sys.modules['asi.reasoning_summary_translator'] = rst_mod
loader.exec_module(rst_mod)
ReasoningSummaryTranslator = rst_mod.ReasoningSummaryTranslator


class TestHistoryTranslation(unittest.TestCase):
    def test_logger_translates(self):
        tr = CrossLingualTranslator(["es"])
        logger = ReasoningHistoryLogger(translator=tr)
        logger.log("start -> end")
        entries = logger.get_history()
        self.assertIsInstance(entries[0][1], dict)
        self.assertEqual(entries[0][1]["translations"]["es"], "[es] start -> end")


class TestSummaryTranslator(unittest.TestCase):
    def test_summary_translation(self):
        logger = ReasoningHistoryLogger()
        logger.log("start -> end")
        tr = CrossLingualTranslator(["es"])
        st = ReasoningSummaryTranslator(tr)
        info = st.summarize(logger)
        self.assertIn("start", info["summary"])
        self.assertTrue(info["translations"]["es"].startswith("[es]"))


class TestSelfReflectionCLITranslations(unittest.TestCase):
    def test_cli_with_translations(self):
        tr = CrossLingualTranslator(["es"])
        logger = ReasoningHistoryLogger(translator=tr)
        logger.log("start -> not start")
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg = Path(tmpdir) / "asi"
            pkg.mkdir()
            repo = Path('.').resolve()
            (pkg / "__init__.py").write_text(
                "import sys\n"
                f"sys.path.append('{repo}')\n"
                f"__path__.append('{(repo / 'src').as_posix()}')\n"
            )
            (pkg / "data_ingest.py").write_text(
                "class CrossLingualTranslator:\n"
                "    def __init__(self, languages):\n"
                "        self.languages = list(languages)\n"
                "    def translate(self, text, lang):\n"
                "        return f'[{lang}] {text}'\n"
                "    def translate_all(self, text):\n"
                "        return {l: self.translate(text, l) for l in self.languages}\n"
            )
            np_pkg = Path(tmpdir) / "numpy"
            np_pkg.mkdir()
            (np_pkg / "__init__.py").write_text("")

            with tempfile.NamedTemporaryFile("w", delete=False) as f:
                json.dump(logger.entries, f)
                fname = f.name
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{tmpdir}:{Path('.').resolve()}"
            try:
                proc = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "from asi.self_reflection import main; main([\"%s\"])"
                        % fname,
                    ],
                    capture_output=True,
                    text=True,
                    env=env,
                    cwd=tmpdir,
                )
                self.assertEqual(proc.returncode, 0, msg=proc.stderr)
                self.assertIn("Inconsistencies", proc.stdout)
                self.assertIn("[es]", proc.stdout)
            finally:
                os.unlink(fname)


if __name__ == "__main__":  # pragma: no cover - test helper
    unittest.main()
