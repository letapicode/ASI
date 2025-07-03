"""Goal-oriented evaluation harness for Plan.md modules."""

from __future__ import annotations

import argparse
import importlib
import asyncio
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Tuple

import numpy as np
import torch


def log_memory_usage() -> float:
    """Return peak GPU memory usage in MB and reset stats."""
    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        torch.cuda.reset_peak_memory_stats()
        return float(mem)
    return 0.0


# ---------------------------------------------------------------------------
# Module discovery
# ---------------------------------------------------------------------------

def parse_modules(plan_path: str | Path = "docs/Plan.md") -> list[str]:
    """Return unique module names mentioned in ``plan_path``."""
    text = Path(plan_path).read_text(encoding="utf-8")
    mods = re.findall(r"`src/([\w_]+)\.py`", text)
    return sorted(set(mods))


# ---------------------------------------------------------------------------
# Individual module evaluators
# ---------------------------------------------------------------------------

def _eval_import_only(module: str) -> Tuple[bool, str]:
    importlib.import_module(f"asi.{module}")
    return True, "imported"


def _eval_moe_router() -> Tuple[bool, str]:
    from asi import moe_router

    router = moe_router.HashRouter(num_experts=8)
    x = torch.randn(2, 16, 4)
    assign = router(x)
    std = router.load_balance_std(assign)
    return std < 0.5, f"load_balance_std={std:.3f}"


def _eval_flash_attention3() -> Tuple[bool, str]:
    from asi import flash_attention3 as fa3

    q = torch.randn(1, 4, 8)
    k = torch.randn(1, 4, 8)
    v = torch.randn(1, 4, 8)
    out = fa3.flash_attention_3(q, k, v)
    return out.shape == q.shape, f"shape={tuple(out.shape)}"


def _eval_scaling_law() -> Tuple[bool, str]:
    from asi.scaling_law import BreakpointScalingLaw

    compute = np.logspace(0, 2, 8)
    loss = 1.0 / np.sqrt(compute)
    model = BreakpointScalingLaw()
    model.fit(compute, loss)
    pred = model.predict(compute)
    err = float(np.abs(pred - loss).mean())
    return err < 1e-3, f"err={err:.2e}"


def _eval_scaling_breakpoint() -> Tuple[bool, str]:
    from asi.scaling_breakpoint import fit_breakpoint

    x = np.array([1, 10, 100, 1000], dtype=float)
    y = np.log10(x) * -0.5 + 2.0
    model = fit_breakpoint(x, y)
    pred = model.predict(x)
    err = float(np.abs(pred - y).mean())
    return err < 1e-3, f"err={err:.2e}"


def _eval_retnet_retention() -> Tuple[bool, str]:
    from asi.retnet_retention import RetNetRetention

    q = torch.randn(1, 4, 8)
    k = torch.randn(1, 4, 8)
    v = torch.randn(1, 4, 8)
    mod = RetNetRetention(num_heads=2)
    out = mod(q, k, v)
    return out.shape == q.shape, f"shape={tuple(out.shape)}"


def _eval_mamba_block() -> Tuple[bool, str]:
    from asi.mamba_block import MambaBlock

    block = MambaBlock(dim=8)
    x = torch.randn(1, 3, 8)
    out = block(x)
    return out.shape == x.shape, f"shape={tuple(out.shape)}"


def _eval_hyena_filter() -> Tuple[bool, str]:
    from asi.hyena_filter import HyenaFilter

    filt = HyenaFilter(filter_length=4)
    x = torch.randn(1, 6, 8)
    out = filt(x)
    return out.shape == x.shape, f"shape={tuple(out.shape)}"


def _eval_streaming_compression() -> Tuple[bool, str]:
    from asi.streaming_compression import StreamingCompressor

    comp = StreamingCompressor(dim=8, compressed_dim=4, capacity=10)
    x = torch.randn(5, 8)
    comp.add(x)
    c = comp.compressed()
    return c.shape[1] == 4, f"compressed_dim={c.shape[1]}"


def _eval_vector_store() -> Tuple[bool, str]:
    from asi.vector_store import VectorStore

    store = VectorStore(dim=4)
    vec = np.ones((2, 4), dtype=np.float32)
    store.add(vec)
    out, _ = store.search(vec[0])
    return out.shape == (2, 4), f"retrieved={out.shape[0]}"


def _eval_hierarchical_memory() -> Tuple[bool, str]:
    from asi.hierarchical_memory import HierarchicalMemory

    mem = HierarchicalMemory(dim=8, compressed_dim=4, capacity=10)
    x = torch.randn(2, 8)
    mem.add(x)
    retrieved, _ = mem.search(x[0], k=1)
    return retrieved.shape[-1] == 8, f"size={len(mem)}"


def _eval_link_slot_attention() -> Tuple[bool, str]:
    from asi.hierarchical_memory import HierarchicalMemory
    from asi.link_slot_attention import LinkSlotAttention

    mem = HierarchicalMemory(dim=8, compressed_dim=4, capacity=5)
    lsa = LinkSlotAttention(mem, dim=8, k_top=1)
    x = torch.randn(1, 3, 8)
    out = lsa(x)
    return out.shape == x.shape, f"shape={tuple(out.shape)}"


def _eval_megabyte_patching() -> Tuple[bool, str]:
    from asi.megabyte_patching import MegaBytePatching

    patcher = MegaBytePatching(patch_size=4, dim=8)
    x = torch.randint(0, 256, (1, 5))
    out = patcher(x)
    return out.size(-1) == 8, f"patches={out.size(1)}"


def _eval_topk_sparse_attention() -> Tuple[bool, str]:
    from asi.topk_sparse_attention import topk_sparse_attention

    q = torch.randn(1, 2, 4)
    k = torch.randn(1, 3, 4)
    v = torch.randn(1, 3, 4)
    out = topk_sparse_attention(q, k, v, k_top=2)
    return out.shape == q.shape, f"shape={tuple(out.shape)}"


def _eval_paper_to_code() -> Tuple[bool, str]:
    from asi.paper_to_code import transpile

    latex = "\\Function{add}{a,b}\n\\Return{a+b}"
    py = transpile(latex)
    return "return" in py, "transpiled"


def _eval_autobench() -> Tuple[bool, str]:
    from asi.autobench import run_autobench
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test_x.py"
        p.write_text(
            "import unittest\n\nclass T(unittest.TestCase):\n    def test_ok(self):\n        self.assertTrue(True)\n\nif __name__=='__main__':\n    unittest.main()\n"
        )
        results = run_autobench(tmp)
    ok = list(results.values())[0].passed
    return ok, "sandboxed"


def _eval_neural_arch_search() -> Tuple[bool, str]:
    """Run a tiny distributed search to verify the module."""
    from asi.neural_arch_search import DistributedArchSearch

    space = {"layers": [1, 2], "hidden": [8, 16]}

    def score(cfg: Dict[str, int]) -> float:
        # Simple additive score favouring more layers and hidden units
        return cfg["layers"] + cfg["hidden"] / 16

    search = DistributedArchSearch(space, score, max_workers=1)
    best, val = search.search(num_samples=4)
    ok = "layers" in best and "hidden" in best
    return ok, f"score={val:.2f}"


def _eval_self_alignment() -> Tuple[bool, str]:
    """Check simple alignment using :class:`DeliberativeAligner`."""
    from asi.deliberative_alignment import DeliberativeAligner

    aligner = DeliberativeAligner("no violence")
    steps = ["We greet the user.", "We provide help."]
    ok = aligner.check(steps)
    return ok, "aligned" if ok else "violations"


def _eval_adversarial_robustness() -> Tuple[bool, str]:
    """Generate a simple adversarial example and verify output."""
    from asi.adversarial_robustness import AdversarialRobustnessSuite

    def model(p: str) -> float:
        return float(len(p))

    suite = AdversarialRobustnessSuite(model)
    adv = suite.generate("hello", ["hello", "hi", "hey"])
    return adv == "hi", f"adv={adv}"


EVALUATORS: Dict[str, Callable[[], Tuple[bool, str]]] = {
    "moe_router": _eval_moe_router,
    "flash_attention3": _eval_flash_attention3,
    "scaling_law": _eval_scaling_law,
    "scaling_breakpoint": _eval_scaling_breakpoint,
    "retnet_retention": _eval_retnet_retention,
    "mamba_block": _eval_mamba_block,
    "hyena_filter": _eval_hyena_filter,
    "streaming_compression": _eval_streaming_compression,
    "vector_store": _eval_vector_store,
    "hierarchical_memory": _eval_hierarchical_memory,
    "link_slot_attention": _eval_link_slot_attention,
    "megabyte_patching": _eval_megabyte_patching,
    "topk_sparse_attention": _eval_topk_sparse_attention,
    "paper_to_code": _eval_paper_to_code,
    "autobench": _eval_autobench,
    "neural_arch_search": _eval_neural_arch_search,
    "self_alignment": _eval_self_alignment,
    "adversarial_robustness": _eval_adversarial_robustness,
}


# ---------------------------------------------------------------------------
# Runner utilities
# ---------------------------------------------------------------------------

def evaluate_modules(modules: Iterable[str]) -> Dict[str, Tuple[bool, str]]:
    """Evaluate ``modules`` and return their status."""
    results: Dict[str, Tuple[bool, str]] = {}
    for mod in modules:
        fn = EVALUATORS.get(mod, lambda: _eval_import_only(mod))
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            passed, info = fn()
            mem = log_memory_usage()
            info = f"{info} gpu={mem:.1f}MB"
        except Exception as exc:  # pragma: no cover - diagnostic path
            passed, info = False, f"error: {exc}"
        results[mod] = (passed, info)
    return results


async def evaluate_modules_async(modules: Iterable[str]) -> Dict[str, Tuple[bool, str]]:
    """Asynchronously evaluate ``modules`` concurrently."""
    loop = asyncio.get_running_loop()
    tasks = []
    for mod in modules:
        fn = EVALUATORS.get(mod, lambda: _eval_import_only(mod))

        async def run_fn(func: Callable[[], Tuple[bool, str]] = fn) -> Tuple[bool, str]:
            try:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                passed, info = await loop.run_in_executor(None, func)
                mem = log_memory_usage()
                return passed, f"{info} gpu={mem:.1f}MB"
            except Exception as exc:  # pragma: no cover - diagnostic path
                return False, f"error: {exc}"

        tasks.append(run_fn())

    results_list = await asyncio.gather(*tasks)
    return {mod: res for mod, res in zip(modules, results_list)}


def format_results(results: Dict[str, Tuple[bool, str]]) -> str:
    total = len(results)
    passed = sum(1 for ok, _ in results.values() if ok)
    lines = [f"Passed {passed}/{total} modules"]
    for mod, (ok, info) in sorted(results.items()):
        status = "PASS" if ok else "FAIL"
        lines.append(f"{mod}: {status} - {info}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate project modules")
    parser.add_argument(
        "--plan", default="docs/Plan.md", help="Path to Plan.md listing modules"
    )
    parser.add_argument(
        "--concurrent",
        action="store_true",
        help="Run evaluations concurrently using asyncio",
    )
    args = parser.parse_args(argv)

    mods = parse_modules(args.plan)
    if args.concurrent:
        results = asyncio.run(evaluate_modules_async(mods))
    else:
        results = evaluate_modules(mods)
    mem = log_memory_usage()
    out = format_results(results)
    out += f"\nGPU memory used: {mem:.1f} MB"
    print(out)


if __name__ == "__main__":  # pragma: no cover
    main()
