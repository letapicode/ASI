#!/usr/bin/env python
"""Summarize dataset lineage and license information."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:  # pragma: no cover - prefer package if available
    from asi.dataset_lineage_manager import DatasetLineageManager
    from asi.license_inspector import LicenseInspector
except Exception:  # pragma: no cover - fallback for tests
    from src.dataset_lineage_manager import DatasetLineageManager  # type: ignore
    from src.license_inspector import LicenseInspector  # type: ignore


def summarize(root: str | Path, fmt: str = "md") -> str:
    """Return a summary of lineage steps and licenses."""
    mgr = DatasetLineageManager(root)
    mgr.load()
    inspector = LicenseInspector()
    licenses: dict[str, bool] = {}
    for p in Path(root).rglob("*.json"):
        try:
            licenses[str(p)] = inspector.inspect(p)
        except Exception:
            continue

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
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Dataset summary")
    parser.add_argument("root", help="Dataset root directory")
    parser.add_argument("--format", choices=["md", "json"], default="md")
    args = parser.parse_args(argv)
    print(summarize(args.root, args.format))


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
