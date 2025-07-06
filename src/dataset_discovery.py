from __future__ import annotations

import sqlite3
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import requests


@dataclass
class DiscoveredDataset:
    """Metadata for a discovered dataset."""

    name: str
    url: str
    source: str
    license: str = ""
    license_text: str = ""
    weight: float = 0.0


def _parse_rss(text: str, source: str) -> List[DiscoveredDataset]:
    root = ET.fromstring(text)
    out: List[DiscoveredDataset] = []
    for item in root.findall('.//item'):
        title = item.findtext('title', default='')
        link = item.findtext('link', default='')
        lic = item.findtext('license', default='')
        out.append(DiscoveredDataset(title, link, source, lic, lic))
    return out


def discover_huggingface(rss_url: str = 'https://huggingface.co/datasets/rss/new') -> List[DiscoveredDataset]:
    """Return datasets listed on HuggingFace via RSS."""
    resp = requests.get(rss_url, timeout=10)
    resp.raise_for_status()
    return _parse_rss(resp.text, 'huggingface')


def discover_kaggle(rss_url: str = 'https://www.kaggle.com/datasets.rss') -> List[DiscoveredDataset]:
    """Return datasets listed on Kaggle via RSS."""
    resp = requests.get(rss_url, timeout=10)
    resp.raise_for_status()
    return _parse_rss(resp.text, 'kaggle')


def store_datasets(
    dsets: Iterable[DiscoveredDataset],
    db_path: str | Path,
    agent: "DatasetQualityAgent | None" = None,
) -> None:
    """Store discovered datasets in ``db_path`` SQLite database."""
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute(
            'CREATE TABLE IF NOT EXISTS datasets ('
            'name TEXT, source TEXT, url TEXT, license TEXT, license_text TEXT,'
            ' weight REAL, UNIQUE(name, source))'
        )
        rows = []
        for d in dsets:
            if agent is not None:
                d.weight = agent.evaluate(d)
            rows.append((d.name, d.source, d.url, d.license, d.license_text, d.weight))
        conn.executemany(
            'INSERT OR REPLACE INTO datasets(name, source, url, license, license_text, weight) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            rows,
        )
    conn.close()


__all__ = [
    'DiscoveredDataset',
    'discover_huggingface',
    'discover_kaggle',
    'store_datasets',
]
