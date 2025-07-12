from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

import requests

import json
from .dataset_discovery import DiscoveredDataset, _parse_rss, store_datasets
from .dataset_summarizer import summarize_dataset
from .dataset_bias_detector import compute_word_freq, bias_score
from .cross_lingual_fairness import CrossLingualFairnessEvaluator
try:  # pragma: no cover - optional dependency
    from .data_ingest import CrossLingualTranslator
except Exception:  # pragma: no cover - missing torch
    from .translator_fallback import CrossLingualTranslator


class StreamingDatasetWatcher:
    """Continuously poll RSS feeds for new datasets."""

    def __init__(self, feeds: Dict[str, str], db_path: str | Path, interval: int = 3600) -> None:
        """Initialize with mapping of ``rss_url -> source`` and database path."""
        self.feeds = feeds
        self.db_path = Path(db_path)
        self.interval = int(interval)
        self.seen: set[tuple[str, str]] = set()

    def _fetch(self, url: str) -> str:
        parsed = urlparse(url)
        if parsed.scheme == "file":
            return Path(parsed.path).read_text()
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.text

    def _analyze_dataset(self, root: Path) -> None:
        """Compute bias and fairness metrics and save a JSON report."""
        files = list(root.rglob("*.txt"))
        freq = compute_word_freq(files, num_workers=os.cpu_count())
        bscore = bias_score(freq)
        langs = [p.name for p in root.iterdir() if p.is_dir()]
        stats = {l: {"1": len(list((root / l).glob("*.txt")))} for l in langs}
        if stats:
            ev = CrossLingualFairnessEvaluator(
                translator=CrossLingualTranslator(langs)
            )
            fairness = ev.evaluate(stats)
        else:
            fairness = {}
        report = {"bias_score": bscore, "fairness": fairness}
        out = root / "pre_ingest_analysis.json"
        try:
            out.write_text(json.dumps(report, indent=2))
        except Exception:
            pass

    def poll_once(self) -> List[DiscoveredDataset]:
        """Poll all feeds once and store any new entries."""
        new: List[DiscoveredDataset] = []
        for rss_url, source in self.feeds.items():
            try:
                text = self._fetch(rss_url)
            except Exception:
                continue
            dsets = _parse_rss(text, source)
            for d in dsets:
                key = (d.name, d.source)
                if key not in self.seen:
                    new.append(d)
                    self.seen.add(key)
        if new:
            store_datasets(new, self.db_path)
            for d in new:
                parsed = urlparse(d.url)
                if parsed.scheme == "file":
                    try:
                        root = Path(parsed.path)
                        self._analyze_dataset(root)
                        summarize_dataset(root)
                    except Exception:
                        pass
        return new

    def watch(self) -> None:
        """Run indefinitely, polling at the configured interval."""
        while True:
            self.poll_once()
            time.sleep(self.interval)


__all__ = ["StreamingDatasetWatcher"]
