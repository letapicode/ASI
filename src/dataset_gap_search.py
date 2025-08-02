from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List
import aiohttp
import requests

from .dataset_lineage import DatasetLineageManager


@dataclass
class CandidateURL:
    url: str
    title: str
    snippet: str
    language: str | None = None
    domain: str | None = None


def _slugify(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text)[:50]


def formulate_gap_queries(
    datasets: Iterable[dict],
    languages: Iterable[str],
    domains: Iterable[str],
) -> List[str]:
    """Return search queries for missing languages or domains."""
    existing_langs = {d.get("language") for d in datasets if d.get("language")}
    existing_domains = {d.get("domain") for d in datasets if d.get("domain")}

    missing_langs = [l for l in languages if l not in existing_langs]
    missing_domains = [d for d in domains if d not in existing_domains]

    queries: set[str] = set()
    for lang in missing_langs:
        queries.add(f"{lang} language dataset")
        for dom in domains:
            queries.add(f"{lang} {dom} dataset")
    for dom in missing_domains:
        queries.add(f"{dom} dataset")
    return sorted(queries)


def search_candidates(
    query: str,
    api_url: str = "https://api.duckduckgo.com/",
    max_results: int = 5,
) -> List[CandidateURL]:
    """Return candidate URLs for ``query`` using a web search API."""
    try:
        resp = requests.get(
            api_url, params={"q": query, "format": "json"}, timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    results: List[CandidateURL] = []
    for item in data.get("results", [])[:max_results]:
        results.append(
            CandidateURL(
                url=item.get("url", ""),
                title=item.get("title", ""),
                snippet=item.get("snippet", ""),
            )
        )
    return results


async def search_candidates_async(
    query: str,
    session: aiohttp.ClientSession,
    api_url: str = "https://api.duckduckgo.com/",
    max_results: int = 5,
) -> List[CandidateURL]:
    """Asynchronously return candidate URLs for ``query``."""
    try:
        async with session.get(api_url, params={"q": query, "format": "json"}, timeout=10) as resp:
            resp.raise_for_status()
            data = await resp.json()
    except Exception:
        return []

    results: List[CandidateURL] = []
    for item in data.get("results", [])[:max_results]:
        results.append(
            CandidateURL(
                url=item.get("url", ""),
                title=item.get("title", ""),
                snippet=item.get("snippet", ""),
            )
        )
    return results


def run_gap_search(
    datasets: Iterable[dict],
    languages: Iterable[str],
    domains: Iterable[str],
    out_dir: str | Path,
    lineage: DatasetLineageManager,
    api_url: str = "https://api.duckduckgo.com/",
) -> List[CandidateURL]:
    """Search the web for missing data and log results."""
    queries = formulate_gap_queries(datasets, languages, domains)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: List[CandidateURL] = []
    for q in queries:
        found = search_candidates(q, api_url=api_url)
        results.extend(found)
        if not found:
            continue
        path = out_dir / f"{_slugify(q)}.json"
        path.write_text(json.dumps([asdict(c) for c in found], indent=2))
        lineage.record([], [path], note=f"gap search: {q}")
    return results


async def run_gap_search_async(
    datasets: Iterable[dict],
    languages: Iterable[str],
    domains: Iterable[str],
    out_dir: str | Path,
    lineage: DatasetLineageManager,
    api_url: str = "https://api.duckduckgo.com/",
) -> List[CandidateURL]:
    """Asynchronously search the web for missing data and log results."""
    queries = formulate_gap_queries(datasets, languages, domains)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: List[CandidateURL] = []
    async with aiohttp.ClientSession() as session:
        tasks = [search_candidates_async(q, session, api_url=api_url) for q in queries]
        query_results = await asyncio.gather(*tasks)

    for q, found in zip(queries, query_results):
        results.extend(found)
        if not found:
            continue
        path = out_dir / f"{_slugify(q)}.json"
        path.write_text(json.dumps([asdict(c) for c in found], indent=2))
        lineage.record([], [path], note=f"gap search: {q}")
    return results


__all__ = [
    "CandidateURL",
    "formulate_gap_queries",
    "search_candidates",
    "search_candidates_async",
    "run_gap_search",
    "run_gap_search_async",
]
