import random
from typing import Any, Iterable, Tuple, List


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


def _load_log(path: str) -> List[tuple[str, float]]:
    """Return list of (action, reward) pairs from ``path``."""
    entries: List[tuple[str, float]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            action = parts[0]
            try:
                reward = float(parts[1])
            except ValueError:
                continue
            entries.append((action, reward))
    return entries


def main(argv: List[str] | None = None) -> None:
    """Train the agent from an action/reward CSV and print the best action."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train Meta-RL agent from an action,reward log"
    )
    parser.add_argument("log", help="Path to CSV file with 'action,reward' lines")
    args = parser.parse_args(argv)

    entries = _load_log(args.log)
    agent = MetaRLRefactorAgent()
    state = 0
    for action, reward in entries:
        next_state = state + 1
        agent.update(state, action, reward, next_state)
        state = next_state

    best = agent.select_action(state)
    print(f"Best action after {len(entries)} steps: {best}")


if __name__ == "__main__":
    main()
