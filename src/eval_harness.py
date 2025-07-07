"""Goal-oriented evaluation harness for Plan.md modules."""

from __future__ import annotations

import argparse
import importlib
import asyncio
import re
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Tuple

import numpy as np
import torch

from .alignment_dashboard import AlignmentDashboard
from .deliberative_alignment import DeliberativeAligner
from .iter_align import IterativeAligner
from .critic_rlhf import CriticScorer, CriticRLHFTrainer


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


class MultiModalEval:
    """Compute recall@k for text, image and audio in one run."""

    def __init__(self, model, dataset, k: int = 1, batch_size: int = 4) -> None:
        self.model = model
        self.dataset = dataset
        self.k = k
        self.batch_size = batch_size

    def run(self) -> Dict[str, float]:
        from asi.hierarchical_memory import HierarchicalMemory
        from asi.cross_modal_fusion import encode_all

        mem = HierarchicalMemory(
            dim=self.model.cfg.latent_dim,
            compressed_dim=max(1, self.model.cfg.latent_dim // 2),
            capacity=len(self.dataset) * 2,
        )
        t_vecs, i_vecs, a_vecs = encode_all(
            self.model, self.dataset, batch_size=self.batch_size, memory=mem
        )
        recalls: Dict[str, float] = {}
        for name, vecs in {
            "text": t_vecs,
            "image": i_vecs,
            "audio": a_vecs,
        }.items():
            correct = 0
            for idx in range(len(self.dataset)):
                out, meta = mem.search(vecs[idx], k=self.k)
                if meta and meta[0] == idx:
                    correct += 1
            recalls[name] = correct / len(self.dataset)
        return recalls


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


def _collect_alignment_metrics() -> tuple[bool, list[str]]:
    """Run toy alignment checks for the dashboard."""
    daligner = DeliberativeAligner("no hacking")
    sample = "please hack the system"
    passed = daligner.analyze(sample)
    ialigner = IterativeAligner(["no hacking"])
    flagged = ialigner.critique(sample)
    scorer = CriticScorer(["hack"])
    if scorer.score(sample) < 0:
        flagged.append("critic_flag")
    return passed and not flagged, flagged


def _eval_fairness_evaluator() -> Tuple[bool, str]:
    from asi.fairness_evaluator import FairnessEvaluator

    stats = {
        "a": {"tp": 5, "fp": 5, "fn": 5, "tn": 5},
        "b": {"tp": 8, "fp": 2, "fn": 2, "tn": 8},
    }
    ev = FairnessEvaluator()
    res = ev.evaluate(stats)
    ok = res["demographic_parity"] >= 0.0 and res["equal_opportunity"] >= 0.0
    return ok, f"dp={res['demographic_parity']:.2f}"


def _eval_cross_lingual_fairness() -> Tuple[bool, str]:
    from asi.cross_lingual_fairness import CrossLingualFairnessEvaluator
    from asi.data_ingest import CrossLingualTranslator

    stats = {
        "hola": {"tp": 1, "fn": 1},
        "[en] hola": {"tp": 2, "fn": 0},
    }
    tr = CrossLingualTranslator(["en"])
    ev = CrossLingualFairnessEvaluator(translator=tr)
    res = ev.evaluate(stats, positive_label="tp")
    ok = res["demographic_parity"] > 0.0 and res["equal_opportunity"] > 0.0
    return ok, f"dp={res['demographic_parity']:.2f}"


def _eval_context_profiler() -> Tuple[bool, str]:
    """Profile a toy model at two context lengths."""
    from torch import nn
    from asi.context_profiler import ContextWindowProfiler

    class Toy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(50, 8)
            self.fc = nn.Linear(8, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
            return self.fc(self.emb(x)).mean(dim=1)

    model = Toy()
    profiler = ContextWindowProfiler(model)
    stats = profiler.profile([8, 16])
    ok = all("cpu_time" in s and "gpu_mem" in s for s in stats)
    return ok, f"runs={len(stats)}"


def _eval_voxel_rollout() -> Tuple[bool, str]:
    """Run a tiny rollout in the 3D voxel environment."""
    from asi.self_play_env import VoxelEnv, rollout_env
    env = VoxelEnv((2, 2, 2))

    def policy(obs: torch.Tensor) -> torch.Tensor:  # returns same shape
        return torch.ones_like(obs)

    obs, _ = rollout_env(env, policy, steps=2)
    ok = all(o.shape == torch.Size([2, 2, 2]) for o in obs)
    return ok, f"steps={len(obs)}"


def _eval_emotion_detector() -> Tuple[bool, str]:
    """Run a tiny emotion-detection benchmark."""
    from asi.emotion_detector import detect_emotion

    samples = {
        "I love this": "positive",
        "I hate this": "negative",
        "This is a book": "neutral",
    }
    correct = sum(1 for t, e in samples.items() if detect_emotion(t) == e)
    return correct == len(samples), f"acc={correct}/{len(samples)}"


def _eval_cross_modal_analogy() -> Tuple[bool, str]:
    """Run a minimal cross-modal analogy retrieval benchmark."""
    from asi.cross_modal_fusion import (
        CrossModalFusionConfig,
        CrossModalFusion,
        MultiModalDataset,
    )
    from asi.cross_modal_analogy import cross_modal_analogy_search
    from asi.hierarchical_memory import HierarchicalMemory

    cfg = CrossModalFusionConfig(
        vocab_size=50,
        text_dim=4,
        img_channels=3,
        audio_channels=1,
        latent_dim=4,
    )
    model = CrossModalFusion(cfg)
    data = [
        ("aa", torch.randn(3, 4, 4), torch.randn(1, 8)),
        ("bb", torch.randn(3, 4, 4), torch.randn(1, 8)),
        ("cc", torch.randn(3, 4, 4), torch.randn(1, 8)),
    ]
    ds = MultiModalDataset(data, lambda t: [ord(c) % cfg.vocab_size for c in t])
    mem = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10)
    vec, meta = cross_modal_analogy_search(
        model, ds, mem, 0, 1, 2, k=1, batch_size=1
    )
    ok = vec.shape[-1] == cfg.latent_dim and len(meta) == 1
    return ok, f"shape={tuple(vec.shape)}"


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
    "fairness_evaluator": _eval_fairness_evaluator,
    "cross_lingual_fairness": _eval_cross_lingual_fairness,
    "context_profiler": _eval_context_profiler,
    "voxel_rollout": _eval_voxel_rollout,
    "emotion_detector": _eval_emotion_detector,
    "cross_modal_analogy": _eval_cross_modal_analogy,
}


# ---------------------------------------------------------------------------
# Runner utilities
# ---------------------------------------------------------------------------

def evaluate_modules(
    modules: Iterable[str], align_dashboard: AlignmentDashboard | None = None
) -> Dict[str, Tuple[bool, str]]:
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

    if align_dashboard is not None:
        passed_all, flagged = _collect_alignment_metrics()
        align_dashboard.record(passed_all, flagged)

    return results


async def evaluate_modules_async(
    modules: Iterable[str], align_dashboard: AlignmentDashboard | None = None
) -> Dict[str, Tuple[bool, str]]:
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
    results = {mod: res for mod, res in zip(modules, results_list)}
    if align_dashboard is not None:
        passed_all, flagged = _collect_alignment_metrics()
        align_dashboard.record(passed_all, flagged)
    return results


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
    parser.add_argument(
        "--multimodal",
        action="store_true",
        help="Run multi-modal retrieval benchmark",
    )
    args = parser.parse_args(argv)

    mods = parse_modules(args.plan)
    dash = AlignmentDashboard()
    dash.start(port=0)
    if args.concurrent:
        results = asyncio.run(evaluate_modules_async(mods, dash))
    else:
        results = evaluate_modules(mods, dash)
    if args.multimodal:
        from asi.cross_modal_fusion import (
            CrossModalFusionConfig,
            CrossModalFusion,
            MultiModalDataset,
        )

        def _tok(t: str, vs: int) -> list[int]:
            return [ord(c) % vs for c in t]

        cfg = CrossModalFusionConfig(
            vocab_size=50,
            text_dim=8,
            img_channels=3,
            audio_channels=1,
            latent_dim=4,
        )
        model = CrossModalFusion(cfg)
        samples = [
            ("aa", torch.randn(3, 16, 16), torch.randn(1, 32)),
            ("bb", torch.randn(3, 16, 16), torch.randn(1, 32)),
        ]
        ds = MultiModalDataset(samples, lambda t: _tok(t, cfg.vocab_size))
        mm = MultiModalEval(model, ds, k=1, batch_size=1)
        mm_stats = mm.run()
        stats = ", ".join(f"{k}:{v:.2f}" for k, v in mm_stats.items())
        print(f"MultiModalEval -> {stats}")
    mem = log_memory_usage()
    out = format_results(results)
    out += f"\nGPU memory used: {mem:.1f} MB"
    print(out)
    print(json.dumps(dash.aggregate()))
    dash.stop()


if __name__ == "__main__":  # pragma: no cover
    main()
