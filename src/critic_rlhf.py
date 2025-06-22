import torch
from typing import Iterable, Sequence


class CriticScorer:
    """Heuristic critic that assigns rewards based on banned phrases."""

    def __init__(self, banned_phrases: Iterable[str] | None = None) -> None:
        self.banned = [p.lower() for p in banned_phrases or ()]

    def score(self, text: str) -> float:
        """Return +1 unless ``text`` contains a banned phrase, else -1."""
        clean = text.lower()
        for phrase in self.banned:
            if phrase and phrase in clean:
                return -1.0
        return 1.0


class CriticRLHFTrainer:
    """Minimal critic-driven RLHF loop using REINFORCE.

    The trainer samples an action from ``model`` given an input tensor, evaluates
    it with ``CriticScorer``, and updates the model so that high-scoring outputs
    become more likely.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        actions: Sequence[str],
        scorer: CriticScorer,
        lr: float = 1e-2,
    ) -> None:
        if not actions:
            raise ValueError("actions must not be empty")
        self.model = model
        self.actions = list(actions)
        self.scorer = scorer
        self.opt = torch.optim.SGD(model.parameters(), lr=lr)

    def sample(self, x: torch.Tensor) -> tuple[int, torch.Tensor]:
        """Return (index, prob) for a sampled action from ``model(x)``."""
        logits = self.model(x)
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        idx = torch.multinomial(probs, 1).item()
        return idx, probs[idx]

    def train_step(self, x: torch.Tensor) -> float:
        """Sample an action, score it, and update the model."""
        idx, prob = self.sample(x)
        reward = self.scorer.score(self.actions[idx])
        loss = -reward * torch.log(prob + 1e-8)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return reward
