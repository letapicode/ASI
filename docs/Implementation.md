# Implementation Notes for S-1 and S-2

This repository includes starter modules for the first two algorithms listed in `docs/Plan.md`.

## S-1 Sparse Mixture-of-Experts Routing

- `src/moe_router.py` provides a hash-based router that activates at most two experts per token.
- The `load_balance_std` method reports the relative standard deviation across experts.
- The `expert_utilization` method returns token counts per expert so you can inspect distribution.

## S-2 FlashAttention-3 Kernel

- `src/flash_attention3.py` wraps the FlashAttention-3 kernel and exposes `_HAS_FLASH3` to
signal availability. If the import fails, the wrapper calls PyTorch's `scaled_dot_product_attention` instead.

These modules are prototypes to facilitate experimentation and benchmarking.

## Load-Balancing Measurement

See `docs/load_balance.md` for a walkthrough on how expert utilization is computed and how to replicate
the measurement. A quick example:

```python
from src.moe_router import HashRouter
import torch

x = torch.randn(4, 512, 256)
router = HashRouter(num_experts=16)
assign = router(x)
print('std:', router.load_balance_std(assign))
print('counts:', router.expert_utilization(assign))
```

## Benchmark Script

`scripts/benchmark_moe.py` offers a minimal example comparing parameter counts and approximate training
FLOPs with and without the MOE router. Run it directly with `python scripts/benchmark_moe.py`. The
output shows the parameter growth and rough FLOP ratio when routing is enabled. Use this as a starting
point for more detailed experiments.

## FlashAttention-3 Integration

`src/flash_attention3.py` attempts to import the FlashAttention-3 CUDA/ROCm kernel.
The module exposes `_HAS_FLASH3` to indicate whether the kernel was imported.
If not, the wrapper falls back to `torch.nn.functional.scaled_dot_product_attention`.
To build the kernel yourself:

1. Install the `flash-attn` package with CUDA visible: `pip install flash-attn --no-binary flash-attn`.
2. Ensure `TORCH_CUDA_ARCH_LIST` matches your GPU architecture.
3. Set the environment variable `FLASH_ATTENTION_FORCE_BUILD=1` to trigger compilation from source if needed.

After installation, the wrapper will automatically call the optimized kernel.

## S-3 Scaling-law Breakpoint Model

The module `src/scaling_law.py` fits a two-regime power law to empirical compute
vs. loss measurements. The `BreakpointScalingLaw` class estimates one slope
before a specified compute threshold and another after it.

```python
from src.scaling_law import BreakpointScalingLaw

compute = [1e15, 3e15, 1e16, 3e16]
loss = [2.0, 1.5, 1.2, 1.1]
model = BreakpointScalingLaw(break_compute=3e15)
model.fit(compute, loss)
pred = model.predict([5e15])
print('predicted loss:', pred[0])
```

Adjust `break_compute` to the knee of your dataset or leave it `None` to split
the data in half. The model performs simple log-log linear regression on each
segment and returns predictions in the original scale.
