#!/usr/bin/env python
"""CLI to generate a simple model card."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from asi.dataset_lineage import DatasetLineageManager
from asi.telemetry import TelemetryLogger
from asi.model_card import ModelCardGenerator


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate model card")
    parser.add_argument("lineage_root", help="Dataset lineage directory")
    parser.add_argument("eval_json", help="Evaluation results JSON file")
    parser.add_argument("--out", default="model_card.md")
    parser.add_argument("--format", choices=["md", "json"], default="md")
    args = parser.parse_args(argv)

    lineage = DatasetLineageManager(args.lineage_root)
    lineage.load()
    telemetry = TelemetryLogger()
    card = ModelCardGenerator(lineage, telemetry, json.loads(Path(args.eval_json).read_text()))
    card.save(args.out, fmt=args.format)
    print(f"Saved model card to {args.out}")


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
