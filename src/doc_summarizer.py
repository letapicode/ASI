from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from pathlib import Path
from typing import Iterable, List

from .auto_dataset_filter import AutoDatasetFilter


def _lm_summarize(text: str, lm: AutoDatasetFilter, top_k: int = 8) -> str:
    """Summarize ``text`` using a simple unigram language model."""
    words = [w.strip(".,()") for w in text.split()]
    if not words:
        return ""
    scores = {w: lm.score(w) for w in words}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    summary_words = [w for w, _ in ranked[:top_k]]
    return " ".join(summary_words)


def _list_objects(mod) -> List[tuple[str, object]]:
    objs = []
    for name, obj in inspect.getmembers(mod):
        if (inspect.isfunction(obj) or inspect.isclass(obj)) and obj.__module__ == mod.__name__:
            objs.append((name, obj))
    return objs


def summarize_module(module: str, search_path: Iterable[str] | None = None) -> str:
    """Return a markdown summary of ``module``."""
    if search_path:
        for p in search_path:
            if p not in sys.path:
                sys.path.insert(0, p)
    mod = importlib.import_module(module)
    objects = _list_objects(mod)
    docs = [inspect.getdoc(obj) or "" for _, obj in objects]
    lm = AutoDatasetFilter()
    lm.fit(docs)
    lines = [f"# {module}", ""]
    for name, obj in objects:
        doc = inspect.getdoc(obj) or ""
        summary = _lm_summarize(doc, lm)
        lines.append(f"## {name}")
        lines.append("")
        lines.append(summary or "No documentation available.")
        lines.append("")
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize module documentation")
    parser.add_argument("module", help="Module to summarize")
    parser.add_argument(
        "--out-dir", default="docs/autodoc", help="Directory to save markdown"
    )
    args = parser.parse_args(argv)

    md = summarize_module(args.module)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.module.replace('.', '_')}.md"
    out_file.write_text(md)
    print(f"Wrote {out_file}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
