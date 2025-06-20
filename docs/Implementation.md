# Implementation Notes for S-1 and S-2

This repository includes starter modules for the first two algorithms listed in `docs/Plan.md`.

## S-1 Sparse Mixture-of-Experts Routing

- `src/moe_router.py` provides two routers:
  - `HashRouter` uses hash-based gating to activate at most two experts per token.
  - `SwitchRouter` employs a learned linear gate and selects the top-k experts.
  Both expose `load_balance_std` and `expert_utilization` to inspect token distribution.
- `src/moe_layer.py` implements a small MoE feed-forward block using these routers.

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

`scripts/benchmark_moe.py` offers a minimal example comparing parameter counts and approximate
training FLOPs with and without the MOE router.

Run it from the project root. By default it uses `HashRouter`; pass `--router switch` to benchmark the learned router:

```bash
python scripts/benchmark_moe.py --router switch
```

Expected output shows the dense and MOE parameter counts along with their ratio and a rough FLOP
ratio:

```
Dense params: 262912
MOE params: 2236672
Param increase: 8.5
FLOP ratio: 8.5
```

`scripts/moe_vs_dense.py` provides a similar toy benchmark implemented as a standalone module. It
contrasts a dense feed-forward model with the MOE version. Pass `--router switch` to use the learned router.

Run it as:

```bash
python scripts/moe_vs_dense.py --router switch
```

The script prints the same parameter counts and a comparable FLOP ratio, labelled `Param ratio`. Both
scripts are starting points for more detailed experiments.

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

`src/scaling_breakpoint.py` fits a piecewise log--log relation to compute versus
loss. Call `fit_breakpoint()` with arrays of parameter sizes and observed losses
to obtain a `BreakpointModel` instance.

```python
from src.scaling_breakpoint import fit_breakpoint

params = [1e7, 5e7, 1e8, 5e8]
loss = [2.0, 1.8, 1.6, 1.3]
model = fit_breakpoint(params, loss)
print('breakpoint:', model.breakpoint)
print('predictions:', model.predict(params))
```

The helper searches over candidate breakpoints and performs linear regression in
log space on either side. The resulting model can forecast loss beyond the
training range.

## C-1 RetNet Retention Kernel

- `src/retnet_retention.py` implements a minimal retention module.
- It sequentially accumulates `k * v` products with a decay factor and multiplies by the query at each step.
- The design follows the linear-time, constant-memory retention described in the
  RetNet paper and serves as a placeholder for more optimised kernels.

## C-2 Mamba State-Space Block

- `src/mamba_block.py` implements a simplified state-space module with a gated recurrent update.
- The block runs in linear time over the sequence and maintains a per-batch hidden state.
- It serves as a minimal reference for experiments targeting the Mamba architecture described in the paper.
