from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Any

import numpy as np
import torch

from .compute_budget_tracker import ComputeBudgetTracker

from .world_model_rl import (
    RLBridgeConfig,
    TransitionDataset,
    train_world_model,
)


@dataclass
class BCIFeedbackConfig:
    """Configuration for converting EEG signals into rewards."""

    reward_scale: float = 1.0
    freq_band: tuple[float, float] = (8.0, 30.0)
    sample_rate: float = 256.0


class BCIFeedbackTrainer:
    """Convert EEG signals to rewards and train a world model."""

    def __init__(self, rl_cfg: RLBridgeConfig, cfg: BCIFeedbackConfig | None = None) -> None:
        self.rl_cfg = rl_cfg
        self.cfg = cfg or BCIFeedbackConfig()

    # --------------------------------------------------
    def signal_to_reward(self, signal: np.ndarray | torch.Tensor) -> float:
        """Return a scalar reward derived from ``signal``.

        Signals are first transformed to the frequency domain. The average
        power in ``cfg.freq_band`` is scaled into a reward value.
        """

        arr = signal.detach().cpu().numpy() if isinstance(signal, torch.Tensor) else signal
        arr = np.asarray(arr).ravel()
        if arr.size == 0:
            return 0.0
        freqs = np.fft.rfftfreq(arr.size, d=1.0 / self.cfg.sample_rate)
        spectrum = np.abs(np.fft.rfft(arr))
        low, high = self.cfg.freq_band
        mask = (freqs >= low) & (freqs <= high)
        if not np.any(mask):
            power = spectrum.mean()
        else:
            power = spectrum[mask].mean()
        return float(power * self.cfg.reward_scale)

    def build_dataset(
        self,
        states: Iterable[torch.Tensor],
        actions: Iterable[int],
        next_states: Iterable[torch.Tensor],
        signals: Iterable[np.ndarray | torch.Tensor],
    ) -> TransitionDataset:
        transitions = []
        for s, a, ns, sig in zip(states, actions, next_states, signals):
            r = self.signal_to_reward(sig)
            transitions.append((s, int(a), ns, r))
        return TransitionDataset(transitions)

    def train(
        self,
        states: Sequence[torch.Tensor],
        actions: Sequence[int],
        next_states: Sequence[torch.Tensor],
        signals: Sequence[np.ndarray | torch.Tensor],
        *,
        run_id: str | None = None,
        budget: "ComputeBudgetTracker | None" = None,
    ) -> Any:
        """Build a dataset from transitions and train a world model."""

        dataset = self.build_dataset(states, actions, next_states, signals)
        return train_world_model(
            self.rl_cfg,
            dataset,
            run_id=run_id or "bci",
            budget=budget,
        )


__all__ = ["BCIFeedbackConfig", "BCIFeedbackTrainer"]
