from __future__ import annotations

from typing import List, Tuple, Dict
import time
import numpy as np

from .adaptive_planner import AdaptivePlanner
from .hierarchical_memory import HierarchicalMemory
from .graph_of_thought import GraphOfThought
from .reasoning_history import ReasoningHistoryLogger


class DebateAgent:
    """Simple agent wrapping a planner, memory and reasoning graph."""

    __slots__ = ("name", "planner", "memory", "graph", "_last_id")

    def __init__(self, name: str, planner: AdaptivePlanner, memory: HierarchicalMemory) -> None:
        self.name = name
        self.planner = planner
        self.memory = memory
        self.graph = GraphOfThought()
        self._last_id: int | None = None

    def respond(self, prompt: str) -> str:
        """Return planner-selected response and log reasoning step."""
        answer = self.planner.best_strategy([prompt])
        ts = time.time()
        node = self.graph.add_step(answer, metadata={"timestamp": ts, "agent": self.name})
        if self._last_id is not None:
            self.graph.connect(self._last_id, node, timestamp=ts)
        self._last_id = node
        self.memory.add(np.random.randn(1, self.memory.dim))
        return answer


class SocraticDebate:
    """Run a Socratic debate between two agents."""

    __slots__ = ("agents", "logger")

    def __init__(
        self,
        agent_a: DebateAgent,
        agent_b: DebateAgent,
        logger: ReasoningHistoryLogger | None = None,
    ) -> None:
        self.agents = {"A": agent_a, "B": agent_b}
        self.logger = logger or ReasoningHistoryLogger()

    def run_debate(self, question: str, rounds: int = 3) -> Tuple[List[Tuple[str, str]], str]:
        """Run ``rounds`` of alternating Q/A starting from ``question``."""
        transcript: List[Tuple[str, str]] = [("Q", question)]
        current = question
        agent_a = self.agents["A"]
        agent_b = self.agents["B"]
        for _ in range(rounds):
            answer = agent_a.respond(current)
            transcript.append(("A", answer))
            current = agent_b.respond("Why " + answer + "?")
            transcript.append(("B", current))
        verdict = transcript[-1][1]
        self.logger.log_debate(transcript, verdict)
        return transcript, verdict

    def graphs(self) -> Dict[str, GraphOfThought]:
        return {name: ag.graph for name, ag in self.agents.items()}


__all__ = ["DebateAgent", "SocraticDebate"]
