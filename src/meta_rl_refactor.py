import random
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ModuleState:
    """Track performance of a code module."""
    name: str
    performance: float = 1.0


class RefactorEnv:
    """Toy environment for the Meta-RL Refactor Agent."""

    def __init__(self, modules: List[str]) -> None:
        self.modules: Dict[str, ModuleState] = {
            m: ModuleState(name=m) for m in modules
        }

    def step(self, module: str, action: str) -> float:
        """Apply an action and return the reward."""
        state = self.modules[module]
        if action == "replace":
            delta = random.uniform(0.05, 0.2)
        elif action == "refactor":
            delta = random.uniform(0.0, 0.1)
        else:  # rollback
            delta = -random.uniform(0.0, 0.05)
        state.performance += delta
        return delta


class MetaRLRefactorAgent:
    """Simple Q-learning agent choosing module actions."""

    ACTIONS = ("replace", "refactor", "rollback")

    def __init__(self, modules: List[str], lr: float = 0.1, gamma: float = 0.9,
                 epsilon: float = 0.1) -> None:
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_values: Dict[str, Dict[str, float]] = {
            m: {a: 0.0 for a in self.ACTIONS} for m in modules
        }

    def choose_action(self, module: str) -> str:
        if random.random() < self.epsilon:
            return random.choice(self.ACTIONS)
        qvals = self.q_values[module]
        return max(qvals, key=qvals.get)

    def update(self, module: str, action: str, reward: float) -> None:
        q = self.q_values[module][action]
        self.q_values[module][action] = q + self.lr * (reward - q)
