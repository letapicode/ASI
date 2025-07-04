from __future__ import annotations

import asyncio
from typing import Iterable

from .graph_of_thought import GraphOfThought
from .multi_agent_coordinator import MultiAgentCoordinator


class CollaborativeHealingLoop:
    """Coordinate self-healing across multiple agents."""

    def __init__(self, coordinator: MultiAgentCoordinator) -> None:
        self.coordinator = coordinator
        self.graph = GraphOfThought()

    def report_anomaly(self, component: str, detail: str) -> int:
        return self.graph.add_step(f"{component}:{detail}")

    async def run(self, components: Iterable[str]) -> None:
        await self.coordinator.schedule_round(components)


__all__ = ["CollaborativeHealingLoop"]
