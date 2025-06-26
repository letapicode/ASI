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

`src/scaling_law.py` defines ``BreakpointScalingLaw`` which fits a piecewise
log--log relation to compute versus loss. Initialize the model with an optional
breakpoint guess and call ``fit()`` with arrays of compute values and observed
losses.

```python
from src.scaling_law import BreakpointScalingLaw

params = [1e7, 5e7, 1e8, 5e8]
loss = [2.0, 1.8, 1.6, 1.3]
model = BreakpointScalingLaw()
model.fit(params, loss)
print('breakpoint:', model.break_compute)
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

## C-3 Hyena Filter

- `src/hyena_filter.py` adds an FFT-based 1D convolution used in Hyena and H\u00b3 models.
- The module exposes a `HyenaFilter` class with a learnable filter.
- Forward pass performs convolution in the frequency domain and returns a tensor
  with the same shape as the input.
- This provides a prototype for the long-range implicit-FFT filtering described
  in the Plan under **C-3**.

## Streaming Compression

- `src/streaming_compression.py` implements a `StreamingCompressor` that keeps a
  reservoir sample of token embeddings and compresses them with a small
  autoencoder.
- This demonstrates the *streaming compression* step toward infinite context,
  keeping the working set roughly logarithmic in sequence length.

## Hierarchical Vector Store

- `src/vector_store.py` provides an in-memory `VectorStore` with dot-product
  similarity search.
- `VectorStore.add()` stores embeddings with optional metadata and
  `search()` returns the top-k nearest vectors.
- The store now supports `save()` and `load()` to persist vectors and metadata
  to a compressed `.npz` file.
- This serves as a minimal prototype for the *hierarchical retrieval* memory
  described in the Plan.
- `src/hierarchical_memory.py` combines `StreamingCompressor` and
  `VectorStore` into a single `HierarchicalMemory` utility. It compresses
  incoming embeddings, stores them in the vector store and returns decoded
  vectors on search. Search results stay on the same device as the query. This
  wires together steps two and three of the infinite-context roadmap.
  `HierarchicalMemory.save()` and `.load()` persist both the compressor state
  and vector store so memory can be restored from disk.

## C-4 MegaByte Patching

- `src/megabyte_patching.py` contains a `MegaBytePatching` module that groups
  byte sequences into fixed-size patches.
- Each patch is embedded with a learnable byte table and projected to a single
  vector, forming the hierarchical representation used in MegaByte models.

## C-5 Top-k Sparse Attention

- `src/topk_sparse_attention.py` defines a `topk_sparse_attention` function that selects the highest-scoring keys per query.
- This provides a lightweight inference-time approximation to full attention for the **C-5** task.

## C-6 RWKV Infinite-Context Loop

- `src/rwkv_loop.py` provides a simplified recurrent block employing the token-shift trick.
- It keeps only a single hidden state per batch, enabling constant-memory training on very long sequences.

## A-1 Paper-to-Code Transpiler

- `src/paper_to_code.py` offers a minimal transpiler from LaTeX pseudo-code to
  Python. The function `transpile()` maps common `\For`, `\If`, and `\State`
  commands to Python syntax and manages indentation.
- A small command line interface is provided via `python -m src.paper_to_code`
  to convert a LaTeX file into a Python script.

## A-2 AutoBench Harness

- `src/autobench.py` runs each test module in its own subprocess to sandbox imports.
- `summarize_results()` returns a concise scoreboard and shows the first few lines of failing output.
- The command line interface `python -m src.autobench` prints this summary for the specified directory.

## A-3 Meta-RL Refactor Agent

- `src/meta_rl_refactor.py` implements a lightweight Q-learning agent that decides whether to
  *replace*, *refactor*, or *rollback* modules based on benchmark feedback.
- The agent exposes `select_action()` for epsilon-greedy exploration and `update()` to adjust
  its Q-table from observed rewards.

## A-4 Quantum Amplitude-Estimation HPO

- `src/quantum_hpo.py` provides a toy hyper-parameter search using a simulated
  quantum amplitude estimation routine.
- Call `QAEHyperparamSearch.search()` with a candidate parameter set and an
  evaluation function that returns `True` on success. The helper repeatedly
  estimates the success probability via `amplitude_estimate()` and returns the
  best performing setting.

## L-1 Collective Constitutional AI

- `src/collective_constitution.py` aggregates crowd-sourced principles into
  actionable rules.
- `derive_rules()` keeps principles supported by at least `min_agreement` users.
- `label_responses()` tags unlabeled text as safe or unsafe depending on rule
  matches.

## L-2 Deliberative Alignment

- `src/deliberative_alignment.py` implements a simple chain-of-thought checker.
- `DeliberativeAligner.check()` validates each reasoning step against a policy
  text and returns `True` only if no rule is violated.

## L-3 Iterative Constitutional Self-Alignment

- `src/iter_align.py` implements a lightweight iterative alignment loop.
- `IterativeAligner.iterate()` repeatedly critiques transcripts against current rules and
  extends the rule set with any flagged lines.
- This demonstrates a toy version of the "IterAlign" process described in the Plan.

## L-4 Critic-in-the-Loop RLHF

- `src/critic_rlhf.py` implements a lightweight trainer that blends human rewards
  with scores from a critic model.
- `CriticRLHF.update()` mixes the two feedback signals using a weighting factor
  and updates action values.
- `select_action()` returns the highest-valued action with optional
  epsilon-greedy exploration.
