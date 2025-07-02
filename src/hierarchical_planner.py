from __future__ import annotations

from typing import Callable, Iterable, Tuple, List

import torch

from .graph_of_thought import GraphOfThought, ThoughtNode
from .world_model_rl import rollout_policy, WorldModel


class HierarchicalPlanner:
    """Compose high-level reasoning with world-model rollouts."""

    def __init__(
        self,
        graph: GraphOfThought,
        world_model: WorldModel,
        policy: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        self.graph = graph
        self.world_model = world_model
        self.policy = policy

    def compose_plan(
        self,
        start: int,
        goal_pred: Callable[[ThoughtNode], bool],
        init_state: torch.Tensor,
        rollout_steps: int = 5,
    ) -> Tuple[List[int], List[torch.Tensor], List[List[float]]]:
        """Return graph path, visited states and rewards."""
        path = self.graph.search(start, goal_pred)
        state = init_state
        states = [state]
        rewards: List[List[float]] = []
        for _ in path:
            sims, r = rollout_policy(self.world_model, self.policy, state, steps=rollout_steps)
            state = sims[-1]
            states.append(state)
            rewards.append(r)
        return path, states, rewards


__all__ = ["HierarchicalPlanner"]
