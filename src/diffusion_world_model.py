import torch
from typing import List, Optional


class DiffusionWorldModel:
    """Lightweight diffusion model generating environment states."""

    def __init__(self, state_dim: int, noise_scale: float = 0.1) -> None:
        self.state_dim = state_dim
        self.noise_scale = noise_scale

    def sample(
        self, init_state: Optional[torch.Tensor] = None, steps: int = 5
    ) -> List[torch.Tensor]:
        """Return a list of diffused states starting from ``init_state``."""
        if init_state is None:
            state = torch.zeros(self.state_dim)
        else:
            state = init_state.clone()
        states: List[torch.Tensor] = []
        for _ in range(steps):
            noise = torch.randn_like(state) * self.noise_scale
            state = state + noise
            states.append(state.clone())
        return states


__all__ = ["DiffusionWorldModel"]
