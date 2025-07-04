from __future__ import annotations

from typing import Callable, Iterable, Tuple, List

import torch

from .graph_of_thought import GraphOfThought, ThoughtNode
from .world_model_rl import rollout_policy, WorldModel
from .graph_neural_reasoner import GraphNeuralReasoner


class HierarchicalPlanner:
    """Compose high-level reasoning with world-model rollouts."""

    def __init__(
        self,
        graph: GraphOfThought,
        world_model: WorldModel,
        policy: Callable[[torch.Tensor], torch.Tensor],
        reasoner: GraphNeuralReasoner | None = None,
    ) -> None:
        self.graph = graph
        self.world_model = world_model
        self.policy = policy
        self.reasoner = reasoner

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

    def query_relation(self, subj: str, obj: str) -> float:
        if self.reasoner is None:
            return 0.0
        return self.reasoner.predict_link(subj, obj)


__all__ = ["HierarchicalPlanner"]
