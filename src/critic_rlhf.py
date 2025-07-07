from typing import Iterable, Sequence

try:
    import torch  # type: ignore
    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch optional
    torch = None  # type: ignore
    _HAS_TORCH = False

import numpy as np


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
        model,
        actions: Sequence[str],
        scorer: CriticScorer,
        lr: float = 1e-2,
    ) -> None:
        if not actions:
            raise ValueError("actions must not be empty")
        self.actions = list(actions)
        self.scorer = scorer
        self.lr = lr
        if _HAS_TORCH and isinstance(model, torch.nn.Module):
            self.model = model
            if getattr(self.model, "bias", None) is None:
                self.model.register_parameter(
                    "bias", torch.nn.Parameter(torch.zeros(len(actions)))
                )
            self.opt = torch.optim.SGD(self.model.parameters(), lr=lr)
            self._use_torch = True
        else:
            # numpy-based linear model: dict with 'weight' and optional 'bias'
            self.model = model
            if "bias" not in self.model:
                self.model["bias"] = np.zeros(len(actions))
            self._use_torch = False

    def sample(self, x) -> tuple[int, float]:
        """Return (index, prob) for a sampled action from the policy."""
        if self._use_torch:
            logits = self.model(torch.ones_like(x))
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            idx = torch.multinomial(probs, 1).item()
            return idx, probs[idx].item()
        logits = x @ self.model["weight"].T
        if self.model.get("bias") is not None:
            logits = logits + self.model["bias"]
        probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = probs / probs.sum(axis=-1, keepdims=True)
        idx = int(np.random.choice(len(probs[0]), p=probs[0]))
        return idx, float(probs[0, idx])

    def train_step(self, x) -> float:
        """Sample an action, score it, and update the model."""
        idx, prob = self.sample(x)
        reward = self.scorer.score(self.actions[idx])
        if self._use_torch:
            loss = -reward * torch.log(torch.tensor(prob) + 1e-8)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        else:
            logits = x @ self.model["weight"].T
            if self.model.get("bias") is not None:
                logits = logits + self.model["bias"]
            probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = probs / probs.sum(axis=-1, keepdims=True)
            target = np.zeros_like(probs)
            target[0, idx] = 1.0
            grad_logits = -reward * (target - probs)
            self.model["weight"] -= self.lr * grad_logits.T @ x
            if self.model.get("bias") is not None:
                self.model["bias"] -= self.lr * grad_logits[0]
        return reward
