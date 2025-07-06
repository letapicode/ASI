from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

import requests

from .dataset_discovery import DiscoveredDataset, _parse_rss, store_datasets
from .dataset_summarizer import summarize_dataset


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
                        summarize_dataset(Path(parsed.path))
                    except Exception:
                        pass
        return new

    def watch(self) -> None:
        """Run indefinitely, polling at the configured interval."""
        while True:
            self.poll_once()
            time.sleep(self.interval)


__all__ = ["StreamingDatasetWatcher"]
