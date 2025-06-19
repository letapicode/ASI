# Implementation Notes for S-1 and S-2

This repository includes starter modules for the first two algorithms listed in `docs/Plan.md`.

## S-1 Sparse Mixture-of-Experts Routing

- `src/moe_router.py` provides a hash-based router that activates at most two experts per token.
- The `load_balance_std` helper reports the relative standard deviation across experts.
- The `token_counts` helper returns how many tokens route to each expert.

## S-2 FlashAttention-3 Kernel

- `src/flash_attention3.py` contains a placeholder for the FlashAttention-3 kernel.
- For now it falls back to PyTorch's `scaled_dot_product_attention` until the fused kernel is linked.

These modules are prototypes to facilitate experimentation and benchmarking.

## Load-Balancing Measurement

See `docs/load_balance.md` for a walkthrough on how expert utilization is computed and how to replicate the measurement.

## Benchmark Script

`scripts/benchmark_moe.py` offers a minimal example comparing parameter counts and
approximate training FLOPs with and without the MOE router. It is a starting point
for more detailed experiments.

## FlashAttention-3 Integration

`src/flash_attention3.py` attempts to import the FlashAttention-3 CUDA/ROCm kernel.
If it is not available, the function falls back to PyTorch's built-in attention.
To build the kernel yourself:

1. Install the `flash-attn` package with CUDA visible: `pip install flash-attn --no-binary flash-attn`.
2. Ensure `TORCH_CUDA_ARCH_LIST` matches your GPU architecture.
3. Set the environment variable `FLASH_ATTENTION_FORCE_BUILD=1` to trigger compilation from source if needed.

After installation, the wrapper will automatically call the optimized kernel.
