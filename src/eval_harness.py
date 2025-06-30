"""Goal-oriented evaluation harness (A-10)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
import torch

from .moe_router import HashRouter
from .hyena_filter import HyenaFilter
from .flash_attention3 import _HAS_FLASH3
from .scaling_breakpoint import fit_breakpoint
from .topk_sparse_attention import topk_sparse_attention


@dataclass
class EvalResult:
    value: float | bool
    target: str
    passed: bool


def _metric_moe() -> EvalResult:
    router = HashRouter(num_experts=16)
    x = torch.randn(2, 64, 32)
    assign = router(x)
    std = router.load_balance_std(assign)
    passed = std <= 0.03
    return EvalResult(float(std), "load-balance std <=0.03", passed)


def _metric_flash3() -> EvalResult:
    return EvalResult(bool(_HAS_FLASH3), "FlashAttention-3 available", bool(_HAS_FLASH3))


def _metric_scaling() -> EvalResult:
    compute = np.logspace(0, 2, 8)
    break_compute = compute[4]
    slope1, intercept1 = -0.5, 1.0
    slope2, intercept2 = -0.3, 0.8
    log_c = np.log10(compute)
    loss = np.where(
        compute <= break_compute,
        intercept1 + slope1 * log_c,
        intercept2 + slope2 * log_c,
    )
    loss = 10 ** loss
    model = fit_breakpoint(compute, loss)
    preds = model.predict(compute)
    rel_err = float(np.mean(np.abs(preds - loss) / loss))
    passed = rel_err < 0.1
    return EvalResult(rel_err, "fit error <10%", passed)


def _metric_hyena() -> EvalResult:
    module = HyenaFilter(filter_length=4)
    x = torch.randn(2, 32, 3, requires_grad=True)
    out = module(x).sum()
    out.backward()
    norm = module.filter.grad.norm().item()
    passed = norm < 2.0
    return EvalResult(norm, "grad norm <2", passed)


def _metric_topk() -> EvalResult:
    q = torch.randn(1, 4, 4)
    k = torch.randn(1, 6, 4)
    v = torch.randn(1, 6, 4)
    full_scores = torch.matmul(q, k.transpose(-1, -2)) / (4 ** 0.5)
    full_attn = torch.softmax(full_scores, dim=-1)
    full = torch.matmul(full_attn, v)
    out = topk_sparse_attention(q, k, v, k_top=k.size(1))
    diff = torch.abs(full - out).max().item()
    passed = diff < 1e-5
    return EvalResult(diff, "top-k matches full <1e-5", passed)


_METRICS: Dict[str, Callable[[], EvalResult]] = {
    "S-1": _metric_moe,
    "S-2": _metric_flash3,
    "S-3": _metric_scaling,
    "C-3": _metric_hyena,
    "C-5": _metric_topk,
}


def collect_metrics() -> Dict[str, EvalResult]:
    results: Dict[str, EvalResult] = {}
    for key, fn in _METRICS.items():
        try:
            results[key] = fn()
        except Exception as exc:  # pragma: no cover
            results[key] = EvalResult(float("nan"), f"error: {exc}", False)
    return results


def format_table(results: Dict[str, EvalResult]) -> str:
    lines = ["ID | Metric | Target | Status", "---|-------|--------|------"]
    for key in sorted(results):
        res = results[key]
        status = "PASS" if res.passed else "FAIL"
        if isinstance(res.value, float):
            val_str = f"{res.value:.4f}"
        else:
            val_str = str(res.value)
        lines.append(f"{key} | {val_str} | {res.target} | {status}")
    return "\n".join(lines)


def main() -> None:
    res = collect_metrics()
    print(format_table(res))


if __name__ == "__main__":
    main()
