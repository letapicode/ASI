import random
from typing import Iterable, Tuple


class CriticRLHF:
    """Simplified RLHF trainer using a critic model (Plan.md L-4)."""

    def __init__(
        self,
        actions: Iterable[str],
        critic_weight: float = 0.5,
        lr: float = 0.1,
        epsilon: float = 0.1,
    ) -> None:
        self.actions = tuple(actions)
        if not self.actions:
            raise ValueError("actions must not be empty")
        self.critic_weight = critic_weight
        self.lr = lr
        self.epsilon = epsilon
        self.values: dict[str, float] = {a: 0.0 for a in self.actions}

    def select_action(self) -> str:
        """Choose an action using an epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return max(self.actions, key=lambda a: self.values.get(a, 0.0))

    def update(self, action: str, human_score: float, critic_score: float) -> float:
        """Update value estimates from human and critic feedback."""
        if action not in self.actions:
            raise ValueError(f"Invalid action: {action}")
        reward = (1 - self.critic_weight) * human_score + self.critic_weight * critic_score
        value = self.values.get(action, 0.0)
        value += self.lr * (reward - value)
        self.values[action] = value
        return value
