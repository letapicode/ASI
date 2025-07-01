"""Graph-of-thought planning combined with meta-RL refactoring."""

from typing import Callable, Iterable, List, Tuple, Any

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
    """Rank strategies using a scoring function."""

    def __init__(self, scorer: Callable[[str], float]) -> None:
        self.scorer = scorer

    def rank(self, strategies: Iterable[str]) -> List[Tuple[str, float]]:
        ranked = [(s, float(self.scorer(s))) for s in strategies]
        ranked.sort(key=lambda x: x[1], reverse=True)
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
