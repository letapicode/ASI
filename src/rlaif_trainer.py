from __future__ import annotations

from typing import Callable, Sequence

try:
    import torch  # type: ignore
    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch optional
    torch = None  # type: ignore
    _HAS_TORCH = False

import numpy as np

from .critic_rlhf import CriticRLHFTrainer


class SyntheticCritic:
    """Synthetic critic that scores text via a local model."""

    def __init__(self, model, tokenizer: Callable[[str], np.ndarray]):
        self.model = model
        self.tokenizer = tokenizer

    def score(self, text: str) -> float:
        """Return +1 if the model prefers the text, else -1."""
        x = self.tokenizer(text).reshape(1, -1).astype(float)
        if _HAS_TORCH and isinstance(self.model, torch.nn.Module):
            self.model.eval()
            with torch.no_grad():
                logits = self.model(torch.tensor(x, dtype=torch.float32))
                prob = torch.softmax(logits, dim=-1)[0]
                return 1.0 if prob[1] > prob[0] else -1.0
        logits = x @ self.model["weight"].T
        if self.model.get("bias") is not None:
            logits = logits + self.model["bias"]
        prob = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        prob = prob / prob.sum(axis=-1, keepdims=True)
        return 1.0 if prob[0, 1] > prob[0, 0] else -1.0


class RLAIFTrainer:
    """Minimal reinforcement learning from AI feedback."""

    def __init__(
        self,
        model,
        actions: Sequence[str],
        critic_model,
        tokenizer: Callable[[str], np.ndarray],
        lr: float = 1e-2,
    ) -> None:
        self.critic = SyntheticCritic(critic_model, tokenizer)
        self.trainer = CriticRLHFTrainer(model, actions, self.critic, lr=lr)

    def train_step(self, x) -> float:
        """Train the policy for one step using AI feedback."""
        return self.trainer.train_step(x)

    def sample(self, x):
        return self.trainer.sample(x)


__all__ = ["RLAIFTrainer", "SyntheticCritic"]
