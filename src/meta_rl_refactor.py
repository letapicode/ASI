import random
from typing import Any, Iterable, Tuple


class MetaRLRefactorAgent:
    """Simple Q-learning agent for Plan.md task A-3."""

    def __init__(
        self,
        actions: Iterable[str] = ("replace", "refactor", "rollback"),
        epsilon: float = 0.1,
        alpha: float = 0.5,
        gamma: float = 0.9,
    ) -> None:
        self.actions = tuple(actions)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q: dict[Tuple[Any, str], float] = {}

    def select_action(self, state: Any) -> str:
        """Choose an action using an epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        qvals = [self.q.get((state, a), 0.0) for a in self.actions]
        max_q = max(qvals)
        for action, qval in zip(self.actions, qvals):
            if qval == max_q:
                return action
        return self.actions[0]

    def update(self, state: Any, action: str, reward: float, next_state: Any) -> None:
        """Update the Q-value table given a transition."""
        if action not in self.actions:
            raise ValueError(f"Invalid action: {action}")
        current = self.q.get((state, action), 0.0)
        next_q = max(self.q.get((next_state, a), 0.0) for a in self.actions)
        target = reward + self.gamma * next_q
        self.q[(state, action)] = current + self.alpha * (target - current)
