import json
import tempfile
import unittest
import asyncio
from pathlib import Path
from unittest.mock import patch
import types
import sys

# Stub out dataset_watermarker to avoid optional Pillow dependency
wm_stub = types.ModuleType("asi.dataset_watermarker")
wm_stub.detect_watermark = lambda path: None
sys.modules["asi.dataset_watermarker"] = wm_stub

from asi.dataset_gap_search import (
    formulate_gap_queries,
    run_gap_search,
    run_gap_search_async,
    CandidateURL,
)
from asi.dataset_lineage_manager import DatasetLineageManager


class TestDatasetGapSearch(unittest.TestCase):
    def test_formulate_gap_queries(self):
        datasets = [
            {"language": "en", "domain": "news"},
            {"language": "fr", "domain": "science"},
        ]
        queries = formulate_gap_queries(datasets, ["en", "de"], ["news", "health"])
        self.assertIn("de language dataset", queries)
        self.assertIn("de news dataset", queries)
        self.assertIn("de health dataset", queries)
        self.assertIn("health dataset", queries)
        self.assertNotIn("en news dataset", queries)

    def test_run_gap_search(self):
        data = [{"language": "en", "domain": "news"}]
        cand = CandidateURL(url="http://x", title="t", snippet="s")
        with tempfile.TemporaryDirectory() as tmp:
            mgr = DatasetLineageManager(tmp)
            with patch(
                "asi.dataset_gap_search.search_candidates", return_value=[cand]
            ):
                res = run_gap_search(
                    data,
                    ["de"],
                    ["health"],
                    tmp,
                    mgr,
                )
            self.assertEqual(len(res), 3)
            out_file = Path(tmp) / "de_health_dataset.json"
            self.assertTrue(out_file.exists())
            log = json.loads((Path(tmp) / "dataset_lineage.json").read_text())
            self.assertEqual(log[0]["note"], "gap search: de health dataset")

    def test_run_gap_search_async(self):
        data = [{"language": "en", "domain": "news"}]
        cand = CandidateURL(url="http://x", title="t", snippet="s")

        async def fake_search(*args, **kwargs):
            return [cand]

        with tempfile.TemporaryDirectory() as tmp:
            mgr = DatasetLineageManager(tmp)
            with patch(
                "asi.dataset_gap_search.search_candidates_async", side_effect=fake_search
            ):
                res = asyncio.run(
                    run_gap_search_async(data, ["de"], ["health"], tmp, mgr)
                )

            self.assertEqual(len(res), 3)
            out_file = Path(tmp) / "de_health_dataset.json"
            self.assertTrue(out_file.exists())
            log = json.loads((Path(tmp) / "dataset_lineage.json").read_text())
            self.assertEqual(log[0]["note"], "gap search: de health dataset")


if __name__ == "__main__":
    unittest.main()
