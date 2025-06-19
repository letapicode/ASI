# ASI Algorithm Prototypes

This repository collects early experiments toward a self-improving Artificial Super-Intelligence (ASI).
The long-term plan and algorithmic gaps are documented in [docs/Plan.md](docs/Plan.md).

## Getting Started

The code in `src/` implements prototype modules for:

- **S-1:** hash-based Mixture-of-Experts routing (`src/moe_router.py`)
- **S-2:** FlashAttention‑3 wrapper with fallback (`src/flash_attention3.py`)
- **S-3:** scaling-law breakpoint fitting (`src/scaling_breakpoint.py` and `src/scaling_law.py`)

Example scripts under `scripts/` demonstrate MOE vs dense model benchmarks:

```bash
python scripts/benchmark_moe.py
python scripts/moe_vs_dense.py
```

A minimal Python test suite lives in `tests/`. Run it with:

```bash
pytest
```

For deeper explanations see [docs/Implementation.md](docs/Implementation.md) and
[docs/load_balance.md](docs/load_balance.md).
