from __future__ import annotations

from typing import Tuple, List

from .multi_agent_coordinator import MultiAgentCoordinator
from .graph_of_thought import GraphOfThought
from .reasoning_merger import merge_graphs


def compute_consensus(
    coordinator: MultiAgentCoordinator,
) -> Tuple[GraphOfThought, List[Tuple[float, Tuple[str, ...]]]]:
    """Return merged reasoning graph and inconsistencies.

    The function collects ``GraphOfThought`` objects from all agents registered
    in ``coordinator``.  It then calls :func:`reasoning_merger.merge_graphs` to
    combine them and detect timestamp conflicts.  Agents are expected to expose
    one of ``graph``, ``reasoning_graph`` or ``got`` attributes.
    """
    graphs: dict[str, GraphOfThought] = {}
    for name, agent in coordinator.agents.items():
        g = getattr(agent, "graph", None) or getattr(agent, "reasoning_graph", None) or getattr(agent, "got", None)
        if isinstance(g, GraphOfThought):
            graphs[name] = g
    if not graphs:
        raise ValueError("No agent graphs found")
    merged, inconsistencies = merge_graphs(graphs)
    return merged, inconsistencies


def report_disagreements(inconsistencies: List[Tuple[float, Tuple[str, ...]]]) -> str:
    """Return a text report summarising ``inconsistencies``."""
    if not inconsistencies:
        return "All agents agree"
    lines = ["Disagreements detected:"]
    for ts, texts in inconsistencies:
        joined = ", ".join(texts)
        lines.append(f"timestamp {ts}: {joined}")
    return "\n".join(lines)


__all__ = ["compute_consensus", "report_disagreements"]
