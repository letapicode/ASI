from typing import Iterable, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def transfer_skills(
    policy: nn.Module,
    demonstrations: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    epochs: int = 5,
    lr: float = 1e-3,
) -> None:
    """Fine-tune ``policy`` on demonstration tuples ``(obs, action)``."""
    optim = torch.optim.Adam(policy.parameters(), lr=lr)
    for _ in range(epochs):
        for obs, action in demonstrations:
            pred = policy(obs)
            loss = F.mse_loss(pred, action)
            optim.zero_grad()
            loss.backward()
            optim.step()

