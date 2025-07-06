"""Graph-of-thought planning combined with meta-RL refactoring."""

from typing import Callable, Iterable, List, Tuple, Any

try:
    from .cross_lingual_graph import CrossLingualReasoningGraph
except Exception:  # pragma: no cover - fallback for direct module load
    import importlib.util as _cg_util
    from pathlib import Path as _Path

    _cg_path = _Path(__file__).resolve().parent / "cross_lingual_graph.py"
    _cg_spec = _cg_util.spec_from_file_location("cross_lingual_graph", _cg_path)
    _cg_mod = _cg_util.module_from_spec(_cg_spec)
    assert _cg_spec and _cg_spec.loader
    _cg_spec.loader.exec_module(_cg_mod)  # type: ignore[attr-defined]
    CrossLingualReasoningGraph = _cg_mod.CrossLingualReasoningGraph

try:  # Allow execution as script without package context
    from .meta_rl_refactor import MetaRLRefactorAgent
except Exception:  # pragma: no cover - fallback for direct module load
    import importlib.util as _ilu
    from pathlib import Path as _Path

    _path = _Path(__file__).resolve().parent / "meta_rl_refactor.py"
    spec = _ilu.spec_from_file_location("meta_rl_refactor", _path)
    module = _ilu.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    MetaRLRefactorAgent = module.MetaRLRefactorAgent


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
    ) -> None:
        self.planner = GraphOfThoughtPlanner(scorer)
        self.agent = MetaRLRefactorAgent(actions=actions, epsilon=epsilon)
        self.state = 0

    def rank_strategies(self, strategies: Iterable[str]) -> List[Tuple[str, float]]:
        ranked = self.planner.rank(strategies)
        next_state = self.state
        for strategy, score in ranked:
            action = strategy.split()[0]
            self.agent.update(self.state, action, score, next_state)
        return ranked

    def best_strategy(self, strategies: Iterable[str]) -> str:
        ranked = self.rank_strategies(strategies)
        action = self.agent.select_action(self.state)
        for strat, _ in ranked:
            if strat.startswith(action):
                return strat
        return ranked[0][0]
