from __future__ import annotations
"""Graph-of-thought planning combined with meta-RL refactoring."""

from typing import Callable, Iterable, List, Tuple, Any
from torch import nn

try:
    from .meta_optimizer import MetaOptimizer
except Exception:  # pragma: no cover - fallback for direct module load
    import importlib.util as _mo_util
    from pathlib import Path as _Path

    _mo_path = _Path(__file__).resolve().parent / "meta_optimizer.py"
    _mo_spec = _mo_util.spec_from_file_location("meta_optimizer", _mo_path)
    _mo_mod = _mo_util.module_from_spec(_mo_spec)
    if "asi" not in globals():
        import types as _types
        globals()["asi"] = _types.ModuleType("asi")
    _mo_mod.__package__ = "asi"
    assert _mo_spec and _mo_spec.loader
    _mo_spec.loader.exec_module(_mo_mod)  # type: ignore[attr-defined]
    MetaOptimizer = _mo_mod.MetaOptimizer

try:
    from .cross_lingual_graph import CrossLingualReasoningGraph
except Exception:  # pragma: no cover - fallback for direct module load
    class CrossLingualReasoningGraph:
        def add_step(self, *args: Any, **kwargs: Any) -> None:
            pass

try:  # Allow execution as script without package context
    from .meta_rl_refactor import MetaRLRefactorAgent
except Exception:  # pragma: no cover - fallback for direct module load
    class MetaRLRefactorAgent:
        def __init__(self, actions: Iterable[str], epsilon: float = 0.1) -> None:
            self.actions = tuple(actions)
            self.epsilon = epsilon
            self.q: dict[tuple[Any, str], float] = {}

        def select_action(self, state: Any) -> str:
            import random
            if random.random() < self.epsilon:
                return random.choice(self.actions)
            qvals = [self.q.get((state, a), 0.0) for a in self.actions]
            max_q = max(qvals)
            for a, q in zip(self.actions, qvals):
                if q == max_q:
                    return a
            return self.actions[0]

        def update(self, state: Any, action: str, reward: float, next_state: Any) -> None:
            self.q[(state, action)] = reward


class GraphOfThoughtPlanner:
    """Rank strategies and log them to an optional reasoning graph."""

    def __init__(
        self,
        scorer: Callable[[str], float],
        graph: "CrossLingualReasoningGraph | None" = None,
        language: str = "en",
    ) -> None:
        self.scorer = scorer
        self.graph = graph
        self.language = language

    def rank(self, strategies: Iterable[str]) -> List[Tuple[str, float]]:
        ranked = [(s, float(self.scorer(s))) for s in strategies]
        ranked.sort(key=lambda x: x[1], reverse=True)
        if self.graph is not None:
            for s, score in ranked:
                self.graph.add_step(s, lang=self.language, metadata={"score": score})
        return ranked


class AdaptivePlanner:
    """Combine :class:`GraphOfThoughtPlanner` with :class:`MetaRLRefactorAgent`."""

    def __init__(
        self,
        scorer: Callable[[str], float],
        actions: Iterable[str] = ("replace", "refactor", "rollback"),
        epsilon: float = 0.1,
        *,
        meta_optimizer: "MetaOptimizer | None" = None,
        model: "nn.Module | None" = None,
    ) -> None:
        self.planner = GraphOfThoughtPlanner(scorer)
        self.agent = MetaRLRefactorAgent(actions=actions, epsilon=epsilon)
        self.state = 0
        self.meta_optimizer = meta_optimizer
        self.model = model

    def rank_strategies(self, strategies: Iterable[str]) -> List[Tuple[str, float]]:
        ranked = self.planner.rank(strategies)
        next_state = self.state
        logs: list[tuple[str, float]] = []
        for strategy, score in ranked:
            action = strategy.split()[0]
            self.agent.update(self.state, action, score, next_state)
            logs.append((strategy, score))
        if self.meta_optimizer is not None and self.model is not None:
            self.meta_optimizer.meta_step(self.model, [logs])
        return ranked

    def best_strategy(self, strategies: Iterable[str]) -> str:
        ranked = self.rank_strategies(strategies)
        action = self.agent.select_action(self.state)
        for strat, _ in ranked:
            if strat.startswith(action):
                return strat
        return ranked[0][0]
