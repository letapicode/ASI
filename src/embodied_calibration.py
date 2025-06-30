from typing import Iterable, Tuple

import torch
from torch import nn


def calibrate(
    model: nn.Module,
    sim_data: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    real_data: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    epochs: int = 5,
    lr: float = 1e-3,
) -> None:
    """Align ``model`` parameters using a mix of simulation and real samples."""
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    paired = list(zip(sim_data, real_data))
    for _ in range(epochs):
        for (s_obs, s_act), (r_obs, r_act) in paired:
            pred_sim = model(s_obs)
            pred_real = model(r_obs)
            loss = (
                torch.nn.functional.mse_loss(pred_sim, s_act)
                + torch.nn.functional.mse_loss(pred_real, r_act)
            ) / 2
            optim.zero_grad()
            loss.backward()
            optim.step()

