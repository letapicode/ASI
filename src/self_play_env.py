import random
from dataclasses import dataclass
from typing import List, Tuple


actions = ["left", "right", "up", "down"]


@dataclass
class Transition:
    state: Tuple[int, int]
    action: str
    reward: float
    next_state: Tuple[int, int]


class GridWorld:
    """Minimal grid world for self-play skill discovery."""

    def __init__(self, size: int = 5) -> None:
        self.size = size
        self.reset()

    def reset(self) -> Tuple[int, int]:
        self.pos = [self.size // 2, self.size // 2]
        return tuple(self.pos)

    def step(self, action: str) -> Tuple[Tuple[int, int], float, bool]:
        if action == "left":
            self.pos[0] -= 1
        elif action == "right":
            self.pos[0] += 1
        elif action == "up":
            self.pos[1] -= 1
        elif action == "down":
            self.pos[1] += 1
        self.pos[0] = max(0, min(self.size - 1, self.pos[0]))
        self.pos[1] = max(0, min(self.size - 1, self.pos[1]))
        reward = -1.0
        done = self.pos[0] == 0 and self.pos[1] == 0
        if done:
            reward = 10.0
        return tuple(self.pos), reward, done


class RandomAgent:
    def select_action(self, state: Tuple[int, int]) -> str:
        return random.choice(actions)


def rollout_env(env: GridWorld, agent: RandomAgent, steps: int = 20) -> List[Transition]:
    """Run agent in env and return transitions."""
    state = env.reset()
    log: List[Transition] = []
    for _ in range(steps):
        act = agent.select_action(state)
        next_state, reward, done = env.step(act)
        log.append(Transition(state, act, reward, next_state))
        state = next_state
        if done:
            break
    return log
