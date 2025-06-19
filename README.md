# ASI Prototype

This repository experiments with algorithms needed for self-improving AI. The big picture lives in
[docs/Plan.md](docs/Plan.md), which outlines scaling-efficiency, long-context, and alignment tasks.

## Prototype Scripts

- `scripts/benchmark_moe.py` measures parameter counts and rough FLOPs with and without the sparse Mixture-of-Experts
  router.
- `scripts/moe_vs_dense.py` offers a similar benchmark as a standalone module for quick comparisons between dense and
  MOE feed-forward layers.

## Setup

1. Use Python 3.10 or newer with PyTorch installed.
2. Optional: `pip install flash-attn` to enable the FlashAttention-3 wrapper in `src/flash_attention3.py`.

Run the scripts directly with `python` to see parameter and FLOP estimates.
