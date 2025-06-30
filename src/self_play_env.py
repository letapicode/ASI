from __future__ import annotations

import numpy as np

__all__ = ["SelfPlayEnv", "rollout_env"]


class SelfPlayEnv:
    """Minimal environment for self-play skill discovery."""

    def __init__(self, obs_dim: int = 4, action_dim: int = 2):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state = np.zeros(obs_dim, dtype=np.float32)

    def reset(self) -> tuple[np.ndarray, dict]:
        self.state = np.zeros(self.obs_dim, dtype=np.float32)
        return self.state.copy(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.state += np.random.randn(self.obs_dim) * 0.1
        reward = 1.0 if action == np.argmax(self.state) % self.action_dim else 0.0
        done = np.linalg.norm(self.state) > 10
        return self.state.copy(), reward, done, False, {}


def rollout_env(env: SelfPlayEnv, policy, steps: int = 10) -> list[float]:
    obs, _ = env.reset()
    rewards = []
    for _ in range(steps):
        action = policy(obs)
        obs, reward, done, _, _ = env.step(action)
        rewards.append(reward)
        if done:
            break
    return rewards
