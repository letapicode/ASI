# Implementation Notes for S-1 and S-2

This repository includes starter modules for the first two algorithms listed in `docs/Plan.md`.

## S-1 Sparse Mixture-of-Experts Routing

- `src/moe_router.py` provides a hash-based router that activates at most two experts per token.
- The `load_balance_std` method reports the relative standard deviation across experts.

## S-2 FlashAttention-3 Kernel

- `src/flash_attention3.py` contains a placeholder for the FlashAttention-3 kernel.
- For now it falls back to PyTorch's `scaled_dot_product_attention` until the fused kernel is linked.

These modules are prototypes to facilitate experimentation and benchmarking.
