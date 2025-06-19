# Implementation Notes for S-1 and S-2

This repository includes starter modules for the first two algorithms listed in `docs/Plan.md`.

## S-1 Sparse Mixture-of-Experts Routing

- `src/moe_router.py` provides a hash-based router that activates at most two experts per token.
- The `load_balance_std` method reports the relative standard deviation across experts.

## S-2 FlashAttention-3 Kernel

- `src/flash_attention3.py` contains a placeholder for the FlashAttention-3 kernel.
- For now it falls back to PyTorch's `scaled_dot_product_attention` until the fused kernel is linked.

These modules are prototypes to facilitate experimentation and benchmarking.

## Load-Balancing Measurement

Call `HashRouter.load_balance_std(assignments)` to compute the relative standard deviation across experts.
This value measures how evenly tokens are distributed and matches the `≤0.03` target in `docs/Plan.md`.
See `docs/load_balance.md` for a step-by-step walkthrough.

## Benchmark Script

`scripts/benchmark_moe.py` compares a dense baseline against a MOE variant using `HashRouter`.
Run `python scripts/benchmark_moe.py` to print parameter counts for both models and
the ratio of approximate training FLOPs. Use this as a quick check that parameter
growth aligns with the `docs/Plan.md` goal of ≤10 × parameters with ≤ 1.3 × FLOPs.

## FlashAttention-3 Integration

`src/flash_attention3.py` tries to import the FlashAttention‑3 CUDA/ROCm kernel.
When the build is missing, it falls back to `torch.nn.functional.scaled_dot_product_attention`.
To compile the kernel yourself:

1. Install `flash-attn` with CUDA visible: `pip install flash-attn --no-binary flash-attn`.
2. Set `TORCH_CUDA_ARCH_LIST` to match your GPU architecture.
3. Optionally export `FLASH_ATTENTION_FORCE_BUILD=1` to trigger a source build.

Once installed, rerun your script and the wrapper will automatically use the optimized kernel.
