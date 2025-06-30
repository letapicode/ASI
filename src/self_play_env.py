import numpy as np
import gym
from gym import spaces
from typing import List, Tuple


class SimpleGrid(gym.Env):
    """Tiny grid world used for self-play."""

    metadata = {"render.modes": ["ansi"]}

    def __init__(self, size: int = 5):
        super().__init__()
        self.size = size
        self.observation_space = spaces.Box(low=0, high=size - 1, shape=(2,), dtype=np.int32)
        self.action_space = spaces.Discrete(4)
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = np.array([0, 0], dtype=np.int32)
        return self.pos.copy(), {}

    def step(self, action: int):
        if action == 0:  # up
            self.pos[0] = np.clip(self.pos[0] - 1, 0, self.size - 1)
        elif action == 1:  # down
            self.pos[0] = np.clip(self.pos[0] + 1, 0, self.size - 1)
        elif action == 2:  # left
            self.pos[1] = np.clip(self.pos[1] - 1, 0, self.size - 1)
        elif action == 3:  # right
            self.pos[1] = np.clip(self.pos[1] + 1, 0, self.size - 1)
        done = bool((self.pos == self.size - 1).all())
        reward = float(done)
        return self.pos.copy(), reward, done, False, {}

    def render(self):
        grid = np.full((self.size, self.size), '.', dtype=str)
        grid[self.pos[0], self.pos[1]] = 'A'
        return '\n'.join(' '.join(row) for row in grid)


class SelfPlayAgent:
    def __init__(self, action_space: spaces.Discrete) -> None:
        self.action_space = action_space

    def act(self, obs: np.ndarray) -> int:
        return self.action_space.sample()


def rollout_env(env: gym.Env, agent: SelfPlayAgent, steps: int) -> Tuple[List[np.ndarray], List[float]]:
    """Run ``steps`` interactions in ``env`` using ``agent``."""
    obs, _ = env.reset()
    observations = [obs]
    rewards: List[float] = []
    for _ in range(steps):
        action = agent.act(obs)
        obs, reward, done, _, _ = env.step(action)
        observations.append(obs)
        rewards.append(float(reward))
        if done:
            break
    return observations, rewards

