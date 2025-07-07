from __future__ import annotations

import datetime
import json
import re
import urllib.request
from pathlib import Path
from typing import List, Dict

from .data_ingest import CrossLingualTranslator


def fetch_recent_papers(max_results: int = 5) -> List[Dict[str, str]]:
    """Fetch recent arXiv paper titles and abstracts (best effort)."""
    url = (
        "https://export.arxiv.org/api/query?search_query=all&start=0&max_results="
        f"{max_results}"
    )
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            text = resp.read().decode("utf-8")
    except Exception:
        return []

    entries = re.findall(r"<entry>(.*?)</entry>", text, re.S)
    papers: List[Dict[str, str]] = []
    for ent in entries[:max_results]:
        title_match = re.search(r"<title>([^<]+)</title>", ent)
        summary_match = re.search(r"<summary>([^<]+)</summary>", ent, re.S)
        title = title_match.group(1).replace("\n", " ").strip() if title_match else ""
        summary = (
            summary_match.group(1).replace("\n", " ").strip() if summary_match else ""
        )
        papers.append({"title": title, "summary": summary})
    return papers


def suggest_modules(papers: List[Dict[str, str]]) -> List[str]:
    modules: List[str] = []
    for p in papers:
        t = p["title"].lower()
        if "reinforcement" in t:
            modules.append("world_model_rl")
        if "memory" in t:
            modules.append("hierarchical_memory")
    return modules


def run_ingestion(
    out_dir: str = "research_logs",
    max_results: int = 5,
    translator: CrossLingualTranslator | None = None,
) -> Path:
    """Fetch papers, optionally translate them and write a daily summary."""
    papers = fetch_recent_papers(max_results)

    if translator is not None:
        for p in papers:
            p["title_translations"] = translator.translate_all(p["title"])
            p["summary_translations"] = translator.translate_all(p.get("summary", ""))

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
