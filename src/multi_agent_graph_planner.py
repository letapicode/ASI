from typing import Iterable, Dict, List, Tuple

from .multi_agent_coordinator import MultiAgentCoordinator
from .adaptive_planner import GraphOfThoughtPlanner
from .meta_rl_refactor import MetaRLRefactorAgent


class MultiAgentGraphPlanner:
    """Collaborative planner building a shared reasoning graph."""

    def __init__(
        self,
        coordinator: MultiAgentCoordinator,
        planner: GraphOfThoughtPlanner,
    ) -> None:
        self.coordinator = coordinator
        self.planner = planner
        self.graph: Dict[str, List[Tuple[str, float]]] = {}

    async def plan(self, repos: Iterable[str], strategies: Iterable[str]) -> Dict[str, List[Tuple[str, float]]]:
        """Assign agents and rank strategies for each repo."""
        await self.coordinator.schedule_round(repos)
        ranked = self.planner.rank(strategies)
        for repo in repos:
            self.graph[repo] = ranked
        return self.graph


__all__ = ["MultiAgentGraphPlanner"]
