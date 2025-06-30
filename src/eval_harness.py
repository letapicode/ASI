from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch

from .moe_router import HashRouter
from .flash_attention3 import _HAS_FLASH3


@dataclass
class EvalResult:
    name: str
    value: Any
    target: Any
    passed: bool


def gather_metrics() -> Dict[str, EvalResult]:
    """Collect simple benchmark metrics."""
    results: Dict[str, EvalResult] = {}

    router = HashRouter(num_experts=8)
    dummy = torch.randn(4, 16, 32)
    assign = router(dummy)
    std = router.load_balance_std(assign)
    results["S-1"] = EvalResult("S-1", std, 0.03, std <= 0.03)

    results["S-2"] = EvalResult("S-2", _HAS_FLASH3, True, bool(_HAS_FLASH3))

    return results


def summarize(results: Dict[str, EvalResult]) -> str:
    lines = ["Algorithm  Result  Target  Pass"]
    for res in results.values():
        lines.append(
            f"{res.name:8} {res.value!s:7} {res.target!s:7} {'PASS' if res.passed else 'FAIL'}"
        )
    return "\n".join(lines)


def main() -> None:
    metrics = gather_metrics()
    print(summarize(metrics))


if __name__ == "__main__":
    main()

__all__ = ["gather_metrics", "summarize"]
