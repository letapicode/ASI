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
from src.moe_router import HashRouter, SwitchRouter
import torch

x = torch.randn(4, 512, 256)
router = SwitchRouter(dim=256, num_experts=16)
assign = router(x)
print('load balance std:', router.load_balance_std(assign))
print('expert counts:', router.expert_utilization(assign))
```

Run this code on representative batches and average the reported standard
deviation. If the value exceeds `0.03`, adjust the routing strategy or increase
the number of experts. The `expert_utilization` call returns the token counts per
expert so you can inspect how evenly tokens are distributed.
