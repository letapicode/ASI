from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Tuple, List

import torch

from .world_model_rl import WorldModel

# type alias for constraint functions
ConstraintFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], bool]


@dataclass
class ConstraintViolation:
    """Represents a violation of a logical constraint at a rollout step."""

    step: int
    message: str


class NeuroSymbolicExecutor:
    """Execute world-model rollouts while checking logical constraints."""

    def __init__(self, model: WorldModel, constraints: Iterable[Tuple[str, ConstraintFn]]):
        self.model = model
        self.constraints = list(constraints)

    def _check_constraints(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        step: int,
    ) -> list[ConstraintViolation]:
        violations: list[ConstraintViolation] = []
        for name, fn in self.constraints:
            if not fn(state, action, next_state):
                violations.append(ConstraintViolation(step=step, message=name))
        return violations

    def rollout(
        self,
        policy: Callable[[torch.Tensor], torch.Tensor],
        init_state: torch.Tensor,
        steps: int = 50,
    ) -> Tuple[list[torch.Tensor], list[float], list[ConstraintViolation]]:
        """Roll out the world model using ``policy`` and track constraint violations."""
        device = next(self.model.parameters()).device
        state = init_state.to(device)
        states: list[torch.Tensor] = []
        rewards: list[float] = []
        violations: list[ConstraintViolation] = []
        with torch.no_grad():
            for step in range(steps):
                action = policy(state)
                next_state, reward = self.model(state, action)
                violations.extend(
                    self._check_constraints(state, action, next_state, step)
                )
                states.append(next_state.cpu())
                rewards.append(float(reward.item()))
                state = next_state
        return states, rewards, violations


__all__ = ["ConstraintViolation", "NeuroSymbolicExecutor"]
