# ASI Prototype

This repository experiments with algorithms needed for self-improving AI. The big picture lives in
[docs/Plan.md](docs/Plan.md), which outlines scaling-efficiency, long-context, and alignment tasks.

## Prototype Scripts

- `scripts/benchmark_moe.py` measures parameter counts and FLOPs. It can use a Mixture-of-Experts router.
- `scripts/moe_vs_dense.py` benchmarks dense versus Mixture-of-Experts feed-forward layers.
- `python -m src.paper_to_code` transpiles LaTeX pseudo-code to Python.
- `python -m src.autobench` runs each test file in isolation and reports a summary.

## Setup

1. Use Python 3.10 or newer with PyTorch installed.
2. Install dependencies with `pip install -r requirements.txt`.
3. Optional: `pip install flash-attn` to enable the FlashAttention-3 wrapper in `src/flash_attention3.py`.
4. Run `pip install -e .` to enable imports from the `asi` package.

Run the scripts directly with `python` to see parameter and FLOP estimates.

## Testing

1. Install requirements: `pip install -r requirements.txt`.
2. Install the package in editable mode: `pip install -e .`.
3. Run tests with `pytest`.
