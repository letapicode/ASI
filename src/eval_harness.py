"""Goal-oriented evaluation harness referenced in docs/Plan.md."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch

from .moe_router import HashRouter


@dataclass
class MetricResult:
    """Simple container for a metric evaluation."""

    name: str
    value: float
    target: float
    passed: bool


def parse_plan_metrics(path: str | Path) -> Dict[str, float]:
    """Parse docs/Plan.md for target metrics."""
    text = Path(path).read_text()
    metrics: Dict[str, float] = {}
    m = re.search(r"load-balance std around \*\*(\d+\.\d+)\*\*", text)
    if m:
        metrics["moe_load_balance_std"] = float(m.group(1))
    return metrics


def evaluate_moe_router() -> float:
    """Return load-balance std for the HashRouter."""
    router = HashRouter(num_experts=8)
    x = torch.randn(2, 512, 32)
    assignments = router(x)
    return router.load_balance_std(assignments)


def run_evaluations() -> Dict[str, MetricResult]:
    """Run all metrics and return results."""
    plan_metrics = parse_plan_metrics(Path(__file__).resolve().parents[1] / "docs/Plan.md")
    results: Dict[str, MetricResult] = {}

    value = evaluate_moe_router()
    target = plan_metrics.get("moe_load_balance_std", float("inf"))
    passed = value <= target * 1.5  # allow some slack
    results["moe_load_balance_std"] = MetricResult(
        name="moe_load_balance_std", value=value, target=target, passed=passed
    )

    return results


def summarize_results(results: Dict[str, MetricResult]) -> str:
    """Return a printable scoreboard string."""
    lines = ["Metric                       Value    Target   Status"]
    passed_count = 0
    for r in results.values():
        status = "PASS" if r.passed else "FAIL"
        if r.passed:
            passed_count += 1
        lines.append(f"{r.name:<28} {r.value:>7.4f} {r.target:>7.4f} {status}")
    lines.append(f"Passed {passed_count}/{len(results)} metrics")
    return "\n".join(lines)


def main() -> None:
    results = run_evaluations()
    print(summarize_results(results))


if __name__ == "__main__":
    main()
