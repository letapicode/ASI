#!/usr/bin/env python
"""Summarize dataset lineage, license information and optionally content."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

try:  # pragma: no cover - prefer package if available
    from asi.dataset_lineage_manager import DatasetLineageManager
    from asi.license_inspector import LicenseInspector
except Exception:  # pragma: no cover - fallback for tests
    from src.dataset_lineage_manager import DatasetLineageManager  # type: ignore
    from src.license_inspector import LicenseInspector  # type: ignore
    from src.dataset_summarizer import summarize_dataset  # type: ignore
else:  # pragma: no cover - package import
    from asi.dataset_summarizer import summarize_dataset


def summarize(root: str | Path, fmt: str = "md", content: bool = False) -> str:
    """Return a summary of lineage, licenses and optional content."""
    mgr = DatasetLineageManager(root)
    mgr.load()
    inspector = LicenseInspector()
    licenses: dict[str, bool] = {}
    for p in Path(root).rglob("*.json"):
        try:
            licenses[str(p)] = inspector.inspect(p)
        except Exception:
            continue

    summaries: list[str] | None = None

    if content:
        summaries = summarize_dataset(root)

    if fmt == "json":
        data = {
            "lineage": [
                {
                    "note": s.note,
                    "inputs": s.inputs,
                    "outputs": s.outputs,
                }
                for s in mgr.steps
            ],
            "licenses": licenses,
        }
        if summaries is not None:
            data["content_summaries"] = summaries
        return json.dumps(data, indent=2)

    lines = ["# Dataset Summary", "", "## Lineage"]
    for step in mgr.steps:
        outputs = ", ".join(step.outputs.keys())
        note = step.note or ""
        lines.append(f"- {note} -> {outputs}")
    lines.append("")
    lines.append("## License Compliance")
    for path, ok in licenses.items():
        status = "OK" if ok else "NOT ALLOWED"
        lines.append(f"- {Path(path).name}: {status}")
    if summaries is not None:
        lines.append("")
        lines.append("## Content Summaries")
        for s in summaries:
            lines.append(f"- {s}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Dataset summary")
    parser.add_argument("root", help="Dataset root directory")
    parser.add_argument("--format", choices=["md", "json"], default="md")
    parser.add_argument("--content", action="store_true", help="Summarize dataset content")
    parser.add_argument(
        "--fairness-report",
        metavar="STATS",
        help="Path to JSON stats for fairness visualization",
    )
    args = parser.parse_args(argv)
    out = summarize(args.root, args.format, args.content)
    if args.content and args.format == "md":
        out_dir = Path("docs/datasets")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{Path(args.root).stem}.md"
        out_file.write_text(out)
        print(f"Wrote {out_file}")
    if args.fairness_report:
        from asi.fairness import FairnessVisualizer

        stats_path = Path(args.fairness_report)
        if not stats_path.is_file():
            stats_path = Path(args.root) / stats_path
        if stats_path.is_file():
            data = json.loads(stats_path.read_text())
            vis = FairnessVisualizer()
            img = vis.to_image(data)
            out_dir = Path("docs/datasets")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{Path(args.root).stem}_fairness.png"
            out_file.write_bytes(base64.b64decode(img.split(",", 1)[1]))
            print(f"Wrote {out_file}")
        else:
            print(f"Fairness stats not found: {stats_path}")
    print(out)


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
