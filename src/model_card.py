from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .dataset_lineage import DatasetLineageManager
from .telemetry import TelemetryLogger


@dataclass
class ModelCardGenerator:
    """Collect basic training information into a model card."""

    lineage: DatasetLineageManager
    telemetry: TelemetryLogger
    eval_results: Mapping[str, Any]
    training_args: Optional[Mapping[str, Any]] = None

    def collect(self) -> Dict[str, Any]:
        card = {
            "dataset_lineage": [
                {
                    "note": s.note,
                    "inputs": s.inputs,
                    "outputs": s.outputs,
                }
                for s in self.lineage.steps
            ],
            "telemetry": self.telemetry.get_stats(),
            "evaluation": dict(self.eval_results),
            "training_args": dict(self.training_args or {}),
        }
        return card

    def to_markdown(self, card: Mapping[str, Any]) -> str:
        md = ["# Model Card", ""]
        md.append("## Dataset Lineage")
        for s in card["dataset_lineage"]:
            md.append(f"- {s['note']} -> {list(s['outputs'].keys())}")
        md.append("")
        md.append("## Telemetry")
        for k, v in card["telemetry"].items():
            md.append(f"- {k}: {v}")
        md.append("")
        md.append("## Evaluation")
        for k, v in card["evaluation"].items():
            md.append(f"- {k}: {v}")
        return "\n".join(md)

    def save(self, path: str | Path, fmt: str = "md") -> None:
        card = self.collect()
        p = Path(path)
        if fmt == "json":
            p.write_text(json.dumps(card, indent=2))
        else:
            p.write_text(self.to_markdown(card))


__all__ = ["ModelCardGenerator"]
