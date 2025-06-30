# ASI Prototype

This repository experiments with algorithms needed for self-improving AI. The big picture lives in
[docs/Plan.md](docs/Plan.md), which outlines scaling-efficiency, long-context, and alignment tasks.

## Prototype Scripts

- `scripts/benchmark_moe.py` measures parameter counts and FLOPs. It can use a Mixture-of-Experts router.
- `scripts/moe_vs_dense.py` benchmarks dense versus Mixture-of-Experts feed-forward layers.
- `python -m src.paper_to_code` transpiles LaTeX pseudo-code to Python.
- `python -m src.autobench` runs each test file in isolation and reports a summary.
- `meta-rl-refactor` parses action/reward logs and suggests the next refactoring step.
- `python -m asi.eval_harness` prints a pass/fail table for key metrics.

Example output:

```text
Metric                       Value    Target   Status
moe_load_balance_std         0.0200   0.0200   PASS
Passed 1/1 metrics
```

Example:

```bash
meta-rl-refactor sample_log.csv
```

## Setup

1. Use Python 3.10 or newer with PyTorch installed.
2. Install dependencies with `pip install -r requirements.txt`.
3. Optional: `pip install flash-attn` to enable the FlashAttention-3 wrapper in `src/flash_attention3.py`.
4. Optional: `pip install faiss-cpu` to enable disk-backed vector storage in `src/vector_store.py`.
5. Run `pip install -e .` to enable imports from the `asi` package.
6. The project runs without these optional packages, but FlashAttention-3 and persistent storage will be disabled.

Run the scripts directly with `python` to see parameter and FLOP estimates.

## Testing

1. Install requirements: `pip install -r requirements.txt`.
2. Install the package in editable mode: `pip install -e .`.
3. Run tests with `pytest`.

## Style

This project imposes no strict line-length limit. The AGENTS guidelines instead emphasize writing modular
code rather than enforcing a specific maximum line length.

## CI

Automated tests run on GitHub Actions. The workflow installs the project in editable mode and executes `pytest`.
See [.github/workflows/test.yml](.github/workflows/test.yml) for details.
