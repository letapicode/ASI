from __future__ import annotations

import asyncio
from typing import Iterable, Mapping, Callable, Awaitable, Any

try:  # pragma: no cover - allow running as standalone module
    from .meta_rl_refactor import MetaRLRefactorAgent
except Exception:  # pragma: no cover - fallback for direct import
    import importlib.util as _ilu
    from pathlib import Path as _Path

    _mrf_path = _Path(__file__).resolve().parent / "meta_rl_refactor.py"
    _spec = _ilu.spec_from_file_location("meta_rl_refactor", _mrf_path)
    _mod = _ilu.module_from_spec(_spec)
    assert _spec and _spec.loader
    _spec.loader.exec_module(_mod)  # type: ignore[attr-defined]
    MetaRLRefactorAgent = _mod.MetaRLRefactorAgent

__all__ = ["MultiAgentCoordinator"]


class MultiAgentCoordinator:
    """Coordinate multiple :class:`MetaRLRefactorAgent` instances.

    The coordinator schedules cooperative refactoring tasks across repositories
    and shares rewards among agents. ``apply_fn`` is an async callable executed
    for each ``(repo, action)`` pair and may perform the actual code change.
    ``reward_fn`` returns a numeric reward after applying the action.
    """

    def __init__(self, agents: Mapping[str, MetaRLRefactorAgent] | None = None) -> None:
        self.agents: dict[str, MetaRLRefactorAgent] = dict(agents or {})
        self.log: list[tuple[str, str, str, float]] = []

    def register(self, name: str, agent: MetaRLRefactorAgent) -> None:
        """Register ``agent`` under ``name``."""
        self.agents[name] = agent

    async def _apply_action(
        self,
        name: str,
        agent: MetaRLRefactorAgent,
        repo: str,
        apply_fn: Callable[[str, str], Awaitable[Any]] | None,
        reward_fn: Callable[[str, str], float] | None,
    ) -> None:
        action = agent.select_action(repo)
        if apply_fn is not None:
            await apply_fn(repo, action)
        reward = reward_fn(repo, action) if reward_fn is not None else 0.0
        agent.update(repo, action, reward, repo)
        self.log.append((name, repo, action, reward))

    async def schedule_round(
        self,
        repos: Iterable[str],
        apply_fn: Callable[[str, str], Awaitable[Any]] | None = None,
        reward_fn: Callable[[str, str], float] | None = None,
    ) -> None:
        """Run one coordination round across ``repos``."""
        tasks = [
            self._apply_action(name, agent, repo, apply_fn, reward_fn)
            for repo in repos
            for name, agent in self.agents.items()
        ]
        if tasks:
            await asyncio.gather(*tasks)

    def train_agents(self) -> None:
        """Train each agent on the collected log entries."""
        for name, agent in self.agents.items():
            entries = [
                (action, reward)
                for n, repo, action, reward in self.log
                if n == name
            ]
            if entries:
                agent.train(entries)


