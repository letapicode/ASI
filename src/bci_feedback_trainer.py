from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Any

import numpy as np
import torch
import types
import math

from .compute_budget_tracker import ComputeBudgetTracker
from .secure_federated_learner import SecureFederatedLearner
from .deliberative_alignment import check_alignment

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
        self.feedback_history: list[list[str]] = []

    # --------------------------------------------------
    def _array_from_signal(self, signal: np.ndarray | torch.Tensor) -> list[float]:
        """Return ``signal`` as a flat list of floats."""
        if hasattr(signal, "detach"):
            try:
                signal = signal.detach()
            except Exception:
                pass
        if hasattr(signal, "cpu") and hasattr(signal, "numpy"):
            try:
                arr = signal.cpu().numpy().ravel().tolist()
                return [float(x) for x in arr]
            except Exception:
                pass
        try:
            flat: list[float] = []
            stack = list(signal)
            while stack:
                x = stack.pop()
                if isinstance(x, (list, tuple)):
                    stack.extend(x)
                else:
                    try:
                        flat.append(float(x))
                    except Exception:
                        pass
            if flat:
                return flat[::-1]
        except Exception:
            pass
        try:
            return [float(signal)]
        except Exception:
            return []

    def signal_to_reward(self, signal: np.ndarray | torch.Tensor) -> float:
        """Return a scalar reward derived from ``signal``.

        Uses ``numpy.fft`` when available; otherwise falls back to a mean-square
        energy heuristic.
        """

        arr = self._array_from_signal(signal)
        if not arr:
            return 0.0
        if hasattr(np, "fft") and hasattr(np.fft, "rfft"):
            freqs = np.fft.rfftfreq(len(arr), d=1.0 / self.cfg.sample_rate)
            spectrum = np.abs(np.fft.rfft(arr))
            low, high = self.cfg.freq_band
            mask = (freqs >= low) & (freqs <= high)
            if not np.any(mask):
                power = float(spectrum.mean())
            else:
                power = float(spectrum[mask].mean())
        else:
            power = sum(x * x for x in arr) / len(arr)
        return float(power * self.cfg.reward_scale)

    def detect_feedback(self, signal: np.ndarray | torch.Tensor) -> list[str]:
        """Return event labels derived from ``signal``."""
        arr = self._array_from_signal(signal)
        if not arr:
            return []

        events: list[str] = []
        if hasattr(np, "fft") and hasattr(np.fft, "rfft"):
            freqs = np.fft.rfftfreq(len(arr), d=1.0 / self.cfg.sample_rate)
            spectrum = np.abs(np.fft.rfft(arr))
            alpha = (freqs >= 8.0) & (freqs <= 12.0)
            gamma = (freqs >= 30.0) & (freqs <= 80.0)
            alpha_p = spectrum[alpha].mean() if np.any(alpha) else 0.0
            gamma_p = spectrum[gamma].mean() if np.any(gamma) else 0.0
            if gamma_p > alpha_p * 1.5:
                events.append("discomfort")
        else:
            peak = max(abs(x) for x in arr)
            mean = sum(arr) / len(arr)
            if peak > 2.0 * abs(mean):
                events.append("discomfort")

        mean = sum(arr) / len(arr)
        variance = sum((x - mean) ** 2 for x in arr) / len(arr)
        std = math.sqrt(variance)
        if mean < -0.5 * std:
            events.append("disagreement")
        return events

    def detect_feedback(self, signal: np.ndarray | torch.Tensor) -> list[str]:
        """Return event labels derived from ``signal``."""
        arr = signal.detach().cpu().numpy() if isinstance(signal, torch.Tensor) else signal
        arr = np.asarray(arr).ravel()
        if arr.size == 0:
            return []
        freqs = np.fft.rfftfreq(arr.size, d=1.0 / self.cfg.sample_rate)
        spectrum = np.abs(np.fft.rfft(arr))
        alpha = (freqs >= 8.0) & (freqs <= 12.0)
        gamma = (freqs >= 30.0) & (freqs <= 80.0)
        alpha_p = spectrum[alpha].mean() if np.any(alpha) else 0.0
        gamma_p = spectrum[gamma].mean() if np.any(gamma) else 0.0
        events: list[str] = []
        if gamma_p > alpha_p * 1.5:
            events.append("discomfort")
        if arr.mean() < -0.5 * arr.std():
            events.append("disagreement")
        return events

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
        dtype = getattr(torch, "float32", float)
        enc = [learner.encrypt(torch.tensor([r], dtype=dtype)) for r in rewards]
        dec = [learner.decrypt(e) for e in enc]
        agg = learner.aggregate(dec)
        try:
            val = float(agg.squeeze().item())
        except Exception:
            if isinstance(agg, list):
                val = float(sum(agg) / len(agg))
            else:
                val = float(agg)
        return val

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
                events: list[str] = []
                for sig in node_sigs:
                    events.extend(self.detect_feedback(sig))
                self.feedback_history.append(events)
                if not check_alignment(events):
                    r = -abs(r)
                dtype = getattr(torch, "float32", float)
                transitions.append((s, int(a), ns, torch.tensor(r, dtype=dtype)))
        else:
            assert signals is not None
            for s, a, ns, sig in zip(states, actions, next_states, signals, strict=True):
                r = self.signal_to_reward(sig)
                events = self.detect_feedback(sig)
                self.feedback_history.append(events)
                if not check_alignment(events):
                    r = -abs(r)
                dtype = getattr(torch, "float32", float)
                transitions.append((s, int(a), ns, torch.tensor(r, dtype=dtype)))
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
