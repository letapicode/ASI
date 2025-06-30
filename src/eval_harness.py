import argparse
import importlib
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import torch

from . import autobench

# --------------------------------------------------
# Helpers to parse docs/Plan.md for success criteria
# --------------------------------------------------

def _parse_plan(path: Path = Path("docs/Plan.md")) -> Dict[str, str]:
    """Return mapping of algorithm ID -> success criterion string."""
    targets: Dict[str, str] = {}
    for line in path.read_text().splitlines():
        if line.startswith("|") and "**" in line:
            parts = [p.strip() for p in line.strip("|\n").split("|")]
            if len(parts) >= 4 and parts[0].startswith("**"):
                alg_id = parts[0].strip(" *")
                criterion = parts[3]
                targets[alg_id] = criterion
    return targets

# --------------------------------------------------
# Metric evaluators per algorithm
# --------------------------------------------------

def _eval_s1() -> Tuple[bool, Dict[str, float]]:
    """Check MoE parameter and FLOP ratios."""
    bench = importlib.import_module("scripts.benchmark_moe")
    dense_p, dense_f = bench.run(num_experts=0)
    moe_p, moe_f = bench.run(num_experts=16, router_type="switch")
    param_ratio = moe_p / dense_p
    flop_ratio = moe_f / dense_f
    passed = param_ratio >= 10 and flop_ratio <= 1.3
    return passed, {"param_ratio": param_ratio, "flop_ratio": flop_ratio}

def _eval_s2() -> Tuple[bool, Dict[str, bool]]:
    mod = importlib.import_module("asi.flash_attention3")
    passed = getattr(mod, "_HAS_FLASH3", False)
    return passed, {"flash3": bool(passed)}

def _eval_s3() -> Tuple[bool, Dict[str, float]]:
    mod = importlib.import_module("asi.scaling_law")
    compute = np.logspace(0, 2, 20)
    break_idx = 10
    break_compute = compute[break_idx]
    slope1, intercept1 = -0.6, 2.0
    slope2, intercept2 = -0.3, 1.0
    log_c = np.log10(compute)
    log_loss = np.where(
        compute < break_compute,
        slope1 * log_c + intercept1,
        slope2 * log_c + intercept2,
    )
    loss = 10 ** log_loss
    model = mod.BreakpointScalingLaw(break_compute=break_compute)
    model.fit(compute, loss)
    preds = model.predict(compute)
    rel_err = float(np.mean(np.abs(preds - loss) / loss))
    passed = rel_err < 0.1
    return passed, {"rel_err": rel_err}

def _eval_s4() -> Tuple[bool, Dict[str, float]]:
    mod = importlib.import_module("asi.lora_quant")
    lin = torch.nn.Linear(8, 8)
    qlayer = mod.LoRAQuantLinear(lin, r=4)
    qlayer.quantize()
    passed = qlayer.qa is not None and qlayer.qa.dtype == torch.int8
    ratio = (qlayer.qa.numel() + qlayer.qb.numel()) / (
        qlayer.lora_a.numel() + qlayer.lora_b.numel()
    )
    return passed, {"compressed_ratio": ratio}

EVALUATORS: Dict[str, Callable[[], Tuple[bool, Dict[str, float]]]] = {
    "S-1": _eval_s1,
    "S-2": _eval_s2,
    "S-3": _eval_s3,
    "S-4": _eval_s4,
}


def evaluate_all() -> Dict[str, Tuple[bool, Dict[str, float]]]:
    results = {}
    for alg_id, fn in EVALUATORS.items():
        try:
            res = fn()
        except Exception:  # pragma: no cover - best effort
            res = (False, {"error": 1.0})
        results[alg_id] = res
    return results


def format_table(targets: Dict[str, str], results: Dict[str, Tuple[bool, Dict[str, float]]]) -> str:
    lines = ["ID | Target | Value | Result", "---|---|---|---"]
    for alg_id, (passed, metrics) in results.items():
        target = targets.get(alg_id, "n/a")
        val = ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
        lines.append(f"{alg_id} | {target} | {val} | {'PASS' if passed else 'FAIL'}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Goal-oriented evaluation harness")
    parser.add_argument(
        "--test-dir", default="tests", help="Directory containing test files for autobench"
    )
    args = parser.parse_args()

    targets = _parse_plan()
    results = evaluate_all()
    print(format_table(targets, results))

    # run autobench and append summary
    bench_results = autobench.run_autobench(args.test_dir)
    print()
    print(autobench.summarize_results(bench_results))


if __name__ == "__main__":
    main()
