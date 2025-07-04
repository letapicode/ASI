from __future__ import annotations

from typing import Iterable, Tuple, Callable, Any, List
import torch

from .causal_graph_learner import CausalGraphLearner
from .neuro_symbolic_executor import NeuroSymbolicExecutor
from .world_model_rl import WorldModel


class CausalReasoner:
    """Plan actions using learned causal graphs."""

    def __init__(self, model: WorldModel, threshold: float = 0.3) -> None:
        self.model = model
        self.learner = CausalGraphLearner(threshold=threshold)

    def build_graph(
        self, transitions: Iterable[Tuple[torch.Tensor, int, torch.Tensor]]
    ) -> None:
        data = [
            (s.numpy(), a, ns.numpy())
            for s, a, ns in transitions
        ]
        self.learner.fit(data)

    def plan(
        self,
        policy: Callable[[torch.Tensor], torch.Tensor],
        init_state: torch.Tensor,
        steps: int = 10,
    ) -> dict[str, Any]:
        executor = NeuroSymbolicExecutor(self.model, [])
        states, rewards, violations = executor.rollout(policy, init_state, steps)
        return {
            "states": states,
            "rewards": rewards,
            "edges": self.learner.edges(),
            "violations": violations,
        }


__all__ = ["CausalReasoner"]
