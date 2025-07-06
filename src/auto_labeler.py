"""Generate weak labels for unlabeled triples using a world model."""

from __future__ import annotations

from typing import Iterable, Tuple, Any, List

import random
import logging

import numpy as np
import torch

from .multimodal_world_model import MultiModalWorldModel


class RLLabelingAgent:
    """Simple Q-learning agent for choosing labels."""

    def __init__(
        self,
        num_labels: int,
        *,
        epsilon: float = 0.1,
        alpha: float = 0.5,
        gamma: float = 0.9,
    ) -> None:
        self.num_labels = num_labels
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.q: dict[tuple[int, int], float] = {}
        self.rewards: List[float] = []

    # ---------------------------------------------------------
    def select(self, state: int) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.num_labels - 1)
        qvals = [self.q.get((state, a), 0.0) for a in range(self.num_labels)]
        max_q = max(qvals)
        for a, qv in enumerate(qvals):
            if qv == max_q:
                return a
        return 0

    # ---------------------------------------------------------
    def update(self, state: int, action: int, reward: float, next_state: int | None) -> None:
        next_q = (
            max(self.q.get((next_state, a), 0.0) for a in range(self.num_labels))
            if next_state is not None
            else 0.0
        )
        cur = self.q.get((state, action), 0.0)
        target = reward + self.gamma * next_q
        self.q[(state, action)] = cur + self.alpha * (target - cur)
        self.rewards.append(reward)

    # ---------------------------------------------------------
    def performance(self) -> float:
        if not self.rewards:
            return 0.0
        return float(sum(self.rewards) / len(self.rewards))

    # ---------------------------------------------------------
    def train(self, log_entries: Iterable[tuple[int, int, float]], cycles: int = 1) -> None:
        entries = list(log_entries)
        for _ in range(cycles):
            for state, action, reward in entries:
                self.update(state, action, reward, state)


class AutoLabeler:
    """Weakly label samples by embedding them with a world model."""

    def __init__(self, model: MultiModalWorldModel, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def _hash_label(self, text: str) -> int:
        vocab = self.model.cfg.vocab_size
        return sum(ord(c) for c in text) % vocab

    def label(self, triples: Iterable[Tuple[str, np.ndarray, Any | None]]) -> list[int]:
        device = next(self.model.parameters()).device
        labels = []
        for text, img, _ in triples:
            try:
                t = torch.tensor(self.tokenizer(text), dtype=torch.long, device=device).unsqueeze(0)
                im = torch.tensor(img, dtype=torch.float32, device=device).unsqueeze(0)
                state = self.model.encode_obs(t, im)
                val = int(state.mean().item() * 1000) % self.model.cfg.vocab_size
            except Exception:
                val = self._hash_label(text)
            labels.append(val)
        return labels


class RLAutoLabeler(AutoLabeler):
    """AutoLabeler with RL-guided label selection."""

    def __init__(
        self,
        model: MultiModalWorldModel,
        tokenizer,
        agent: RLLabelingAgent | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(model, tokenizer)
        self.agent = agent or RLLabelingAgent(model.cfg.vocab_size)
        self.logger = logger or logging.getLogger(__name__)

    # -----------------------------------------------------
    def label_with_feedback(
        self,
        triples: Iterable[Tuple[str, np.ndarray, Any | None]],
        biases: Iterable[float],
        feedback: Iterable[float] | None = None,
    ) -> list[int]:
        base_labels = super().label(triples)
        fb_iter = list(feedback or [0.0] * len(base_labels))
        out: list[int] = []
        for state, bias, fb in zip(base_labels, biases, fb_iter):
            action = self.agent.select(state)
            reward = -abs(bias) + float(fb)
            self.agent.update(state, action, reward, state)
            out.append(action)
        self.logger.info("RL labeler perf %.4f", self.agent.performance())
        return out

    # -----------------------------------------------------
    def update_from_logs(
        self, entries: Iterable[tuple[int, int, float]], cycles: int = 1
    ) -> None:
        data = list(entries)
        self.agent.train(data, cycles=cycles)
        self.logger.info("RL labeler updated: %d steps", len(data))


__all__ = ["AutoLabeler", "RLLabelingAgent", "RLAutoLabeler"]
