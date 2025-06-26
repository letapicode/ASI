# ASI Prototype

This repository experiments with algorithms needed for self-improving AI. The big picture lives in
[docs/Plan.md](docs/Plan.md), which outlines scaling-efficiency, long-context, and alignment tasks.

## Prototype Scripts

- `scripts/benchmark_moe.py` measures parameter counts and rough FLOPs with and without the sparse Mixture-of-Experts router.
- `scripts/moe_vs_dense.py` offers a similar benchmark for quick comparisons between dense and MOE feed-forward layers.
- `python -m src.paper_to_code` transpiles LaTeX pseudo-code to Python.
- `python -m src.autobench` runs each test file in isolation and reports a summary.

## Setup

1. Use Python 3.10 or newer with PyTorch installed.
2. Optional: `pip install flash-attn` to enable the FlashAttention-3 wrapper in `src/flash_attention3.py`.
3. Install the package in editable mode with `pip install -e .`.

Run the scripts directly with `python` to see parameter and FLOP estimates.
