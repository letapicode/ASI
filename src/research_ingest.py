from __future__ import annotations

import datetime
import json
import re
import urllib.request
from pathlib import Path
from typing import List, Dict


def fetch_recent_papers(max_results: int = 5) -> List[Dict[str, str]]:
    """Fetch recent arXiv paper titles (best effort)."""
    url = (
        "https://export.arxiv.org/api/query?search_query=all&start=0&max_results="
        f"{max_results}"
    )
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            text = resp.read().decode("utf-8")
    except Exception:
        return []
    titles = re.findall(r"<title>([^<]+)</title>", text)[1:]
    return [{"title": t} for t in titles[:max_results]]


def suggest_modules(papers: List[Dict[str, str]]) -> List[str]:
    modules: List[str] = []
    for p in papers:
        t = p["title"].lower()
        if "reinforcement" in t:
            modules.append("world_model_rl")
        if "memory" in t:
            modules.append("hierarchical_memory")
    return modules


def run_ingestion(out_dir: str, max_results: int = 5) -> Path:
    papers = fetch_recent_papers(max_results)
    summary = {
        "date": str(datetime.date.today()),
        "papers": papers,
        "suggestions": suggest_modules(papers),
    }
    path = Path(out_dir) / f"{summary['date']}.json"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))
    return path


__all__ = ["fetch_recent_papers", "suggest_modules", "run_ingestion"]
