# Measuring Expert Load Balance

This note explains how to evaluate the utilization of experts in the sparse
Mixture-of-Experts router implemented in `src/moe_router.py`.

1. **Collect assignments** – run the router on a batch of tokens to obtain the expert indices for each token.
2. **Count occurrences** – use `torch.bincount` on the flattened assignments to
   tally how many tokens route to each expert.
3. **Compute standard deviation** – calculate the standard deviation of these
   counts and divide by the mean to obtain the relative load imbalance.
4. **Target threshold** – the Plan specifies that the relative standard deviation should remain below `0.03` (3 %).

To reproduce the measurement:

```python
from src.moe_router import HashRouter
import torch

x = torch.randn(4, 512, 256)
router = HashRouter(num_experts=16)
assign = router(x)
print('load balance std:', router.load_balance_std(assign))
```

Run this code on representative batches and average the reported standard
deviation. If the value exceeds `0.03`, adjust the routing strategy or increase
the number of experts.

## Toy example using `HashRouter`

The snippet below walks through a minimal example on a tiny batch.

```python
from src.moe_router import HashRouter
import torch

# Step 1: build the router
router = HashRouter(num_experts=4)

# Step 2: create a toy input with two tokens
x = torch.tensor([[[1., 2.], [3., 4.]]])

# Step 3: route the tokens to experts
assign = router(x)

# Step 4: count how many tokens each expert receives
counts = torch.bincount(assign.view(-1), minlength=router.num_experts)

# Step 5: compute the relative standard deviation of the counts
imbalance = counts.float().std() / counts.float().mean()
print('assignments:', assign)
print('load balance std:', imbalance.item())
```

`docs/Plan.md` sets a 3 % threshold on this standard deviation for stable
training. The toy example above prints both the raw assignments and the computed
imbalance so you can verify the calculation step by step.
