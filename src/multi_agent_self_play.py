from __future__ import annotations

"""Run multiple MetaRLRefactorAgent instances in a self-play environment."""

import asyncio
from dataclasses import dataclass
from typing import Dict

import torch

from .self_play_env import SimpleEnv
from .meta_rl_refactor import MetaRLRefactorAgent
from .multi_agent_coordinator import MultiAgentCoordinator, RLNegotiator
from .multi_agent_dashboard import MultiAgentDashboard


@dataclass
class MultiAgentSelfPlayConfig:
    """Configuration for :func:`run_multi_agent_self_play`."""

    num_agents: int = 2
    env_state_dim: int = 4
    steps: int = 10


class MultiAgentSelfPlay:
    """Wrapper managing agents, environments and telemetry."""

    def __init__(self, cfg: MultiAgentSelfPlayConfig) -> None:
        self.cfg = cfg
        self.agents: Dict[str, MetaRLRefactorAgent] = {
            f"agent{i}": MetaRLRefactorAgent() for i in range(cfg.num_agents)
        }
        self.envs: Dict[str, SimpleEnv] = {
            name: SimpleEnv(cfg.env_state_dim) for name in self.agents
        }
        self.negotiator = RLNegotiator()
        self.coordinator = MultiAgentCoordinator(self.agents, self.negotiator)
        self.dashboard = MultiAgentDashboard(self.coordinator)
        actions = list(next(iter(self.agents.values())).actions)
        self.action_map = {
            act: torch.full((cfg.env_state_dim,), float(i))
            for i, act in enumerate(actions)
        }
        self._last_reward: Dict[str, float] = {}

    async def _apply(self, env_name: str, action: str) -> None:
        env = self.envs[env_name]
        step = env.step(self.action_map[action])
        self._last_reward[env_name] = step.reward

    def _reward(self, env_name: str, action: str) -> float:
        return self._last_reward.get(env_name, 0.0)

    async def run(self) -> None:
        self.dashboard.start(port=0)
        tasks = list(self.envs.keys())
        for env in self.envs.values():
            env.reset()
        for _ in range(self.cfg.steps):
            await self.coordinator.schedule_round(tasks, self._apply, self._reward)
        self.coordinator.train_agents()
        self.dashboard.stop()


def run_multi_agent_self_play(cfg: MultiAgentSelfPlayConfig) -> MultiAgentDashboard:
    """Run self-play with ``cfg`` and return the dashboard."""
    msp = MultiAgentSelfPlay(cfg)
    asyncio.run(msp.run())
    return msp.dashboard


__all__ = ["MultiAgentSelfPlayConfig", "MultiAgentSelfPlay", "run_multi_agent_self_play"]

