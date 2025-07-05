from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Iterable, Tuple, List
import random
import torch

from .robot_skill_transfer import VideoPolicyDataset
from .self_play_env import PrioritizedReplayBuffer


@dataclass
class CurriculumConfig:
    """Configuration for :class:`AdaptiveCurriculum`."""

    lr: float = 0.1


class AdaptiveCurriculum:
    """Blend curated data with self-play logs using a simple bandit."""

    def __init__(
        self,
        curated: VideoPolicyDataset,
        buffer: PrioritizedReplayBuffer,
        cfg: CurriculumConfig | None = None,
    ) -> None:
        self.curated = curated
        self.buffer = buffer
        self.cfg = cfg or CurriculumConfig()
        self.prefs = torch.zeros(2, dtype=torch.float32)
        self.baseline = 0.0

    def _probs(self) -> torch.Tensor:
        return torch.softmax(self.prefs, dim=0)

    def sample(self, batch_size: int) -> Tuple[List[torch.Tensor], List[int], int]:
        """Return batch of (frames, actions) and chosen dataset index."""
        idx = torch.multinomial(self._probs(), 1).item()
        if idx == 0:
            indices = random.sample(range(len(self.curated)), min(batch_size, len(self.curated)))
            frames, actions = zip(*[self.curated[i] for i in indices])
        else:
            frames, actions = self.buffer.sample(batch_size)
        return list(frames), list(actions), idx

    def update(self, idx: int, reward: float) -> None:
        """Update selection preferences from reward."""
        probs = self._probs()
        self.baseline += self.cfg.lr * (reward - self.baseline)
        advantage = reward - self.baseline
        for i in range(2):
            grad = advantage * ((1 if i == idx else 0) - probs[i])
            self.prefs[i] += self.cfg.lr * grad


class SampleWeightRL:
    """REINFORCE loop to adapt sample weights during training."""

    def __init__(self, num_samples: int, lr: float = 0.1) -> None:
        self.prefs = torch.zeros(num_samples, dtype=torch.float32)
        self.lr = lr
        self.baseline = 0.0

    def weights(self) -> torch.Tensor:
        return torch.softmax(self.prefs, dim=0)

    def update(self, idx: int, reward: float) -> None:
        probs = self.weights()
        self.baseline += self.lr * (reward - self.baseline)
        adv = reward - self.baseline
        for i in range(len(self.prefs)):
            grad = adv * ((1 if i == idx else 0) - probs[i])
            self.prefs[i] += self.lr * grad


__all__ = ["CurriculumConfig", "AdaptiveCurriculum", "SampleWeightRL"]
