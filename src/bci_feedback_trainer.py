from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Any

import numpy as np
import torch

from .compute_budget_tracker import ComputeBudgetTracker
from .secure_federated_learner import SecureFederatedLearner

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
        return float(np.float32(power * self.cfg.reward_scale))

    def aggregate_signal_rewards(
        self,
        signals: Iterable[np.ndarray | torch.Tensor],
        learner: SecureFederatedLearner,
    ) -> float:
        """Average rewards from multiple EEG signals securely.

        Each signal is converted into a reward, encrypted and aggregated via
        ``SecureFederatedLearner`` to mimic federated processing.
        """

        rewards = [self.signal_to_reward(sig) for sig in signals]
        enc = [learner.encrypt(torch.tensor([r], dtype=torch.float32)) for r in rewards]
        dec = [learner.decrypt(e) for e in enc]
        agg = learner.aggregate(dec)
        return float(agg.squeeze().item())

    def build_dataset(
        self,
        states: Iterable[torch.Tensor],
        actions: Iterable[int],
        next_states: Iterable[torch.Tensor],
        signals: Iterable[np.ndarray | torch.Tensor] | None,
        *,
        signals_nodes: Sequence[Iterable[np.ndarray | torch.Tensor]] | None = None,
        learner: SecureFederatedLearner | None = None,
    ) -> TransitionDataset:
        """Convert transition tuples into a training dataset.

        When ``signals_nodes`` is provided, rewards are derived by securely
        aggregating signals from each node. Otherwise ``signals`` supplies the
        EEG data for each transition.
        """

        transitions = []
        if signals_nodes is not None:
            learner = learner or SecureFederatedLearner()
            for s, a, ns, node_sigs in zip(states, actions, next_states, zip(*signals_nodes), strict=True):
                r = self.aggregate_signal_rewards(node_sigs, learner)
                transitions.append((s, int(a), ns, torch.tensor(r, dtype=torch.float32)))
        else:
            assert signals is not None
            for s, a, ns, sig in zip(states, actions, next_states, signals, strict=True):
                r = self.signal_to_reward(sig)
                transitions.append((s, int(a), ns, torch.tensor(r, dtype=torch.float32)))
        return TransitionDataset(transitions)

    def train(
        self,
        states: Sequence[torch.Tensor],
        actions: Sequence[int],
        next_states: Sequence[torch.Tensor],
        signals: Sequence[np.ndarray | torch.Tensor] | None,
        *,
        run_id: str | None = None,
        budget: "ComputeBudgetTracker | None" = None,
        signals_nodes: Sequence[Sequence[np.ndarray | torch.Tensor]] | None = None,
        learner: SecureFederatedLearner | None = None,
    ) -> Any:
        """Build a dataset from transitions and train a world model."""

        dataset = self.build_dataset(
            states,
            actions,
            next_states,
            signals,
            signals_nodes=signals_nodes,
            learner=learner,
        )
        return train_world_model(
            self.rl_cfg,
            dataset,
            run_id=run_id or "bci",
            budget=budget,
        )


__all__ = ["BCIFeedbackConfig", "BCIFeedbackTrainer"]
