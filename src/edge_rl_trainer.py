from __future__ import annotations
from typing import Iterable, List, Tuple, Sequence

import numpy as np
from contextlib import nullcontext

import torch
from .analog_backend import AnalogAccelerator

from .compute_budget_tracker import ComputeBudgetTracker
from .adaptive_micro_batcher import AdaptiveMicroBatcher
from .bci_feedback_trainer import BCIFeedbackTrainer


class EdgeRLTrainer:
    """Run world-model updates with compute budget checks."""

    def __init__(
        self,
        model,
        optimizer,
        budget: ComputeBudgetTracker,
        run_id: str = "edge",
        micro_batcher: AdaptiveMicroBatcher | None = None,
        bci_trainer: BCIFeedbackTrainer | None = None,
        *,
        use_loihi: bool = False,
        use_fpga: bool = False,
        use_analog: bool = False,
    ) -> None:
        self.model = model
        self.opt = optimizer
        self.budget = budget
        self.run_id = run_id
        self.micro_batcher = micro_batcher
        self.bci_trainer = bci_trainer
        self.use_loihi = use_loihi
        self.use_fpga = use_fpga
        self.use_analog = use_analog
        self.power_usage: dict[str, float] = {"cpu": 0.0, "loihi": 0.0, "fpga": 0.0, "analog": 0.0}

    def train(
        self,
        data: Iterable[tuple[torch.Tensor, torch.Tensor]],
        threshold: float = 0.1,
    ) -> int:
        steps = 0
        micro = self.micro_batcher
        telemetry = self.budget.telemetry
        start_energy = telemetry.get_stats().get("energy_kwh", 0.0)
        batches: Iterable[List[Tuple[torch.Tensor, torch.Tensor]]]
        if micro is not None:
            micro.start()
            batches = micro.micro_batches(data)
        else:
            batches = [[p] for p in data]
        context = AnalogAccelerator() if self.use_analog else nullcontext()
        with context:
            for batch in batches:
                if self.budget.remaining(self.run_id) <= threshold:
                    break
                states = torch.cat([s.unsqueeze(0) for s, _ in batch], dim=0)
                targets = torch.cat([t.unsqueeze(0) for _, t in batch], dim=0)
                pred = self.model(states)
                loss = torch.nn.functional.mse_loss(pred, targets)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                steps += 1
                if micro is not None:
                    micro.tick()

        if micro is not None:
            micro.stop()
        end_energy = telemetry.get_stats().get("energy_kwh", start_energy)
        delta = end_energy - start_energy
        if self.use_loihi:
            key = "loihi"
        elif self.use_fpga:
            key = "fpga"
        elif self.use_analog:
            key = "analog"
        else:
            key = "cpu"
        self.power_usage[key] += max(delta, 0.0)
        return steps

    # --------------------------------------------------
    def interactive_session(
        self,
        states: Sequence[torch.Tensor],
        actions: Sequence[int],
        next_states: Sequence[torch.Tensor],
        signals: Sequence[torch.Tensor | np.ndarray],
    ):
        """Train a world model from BCI feedback during interactive sessions."""

        if self.bci_trainer is None:
            raise ValueError("bci_trainer not provided")
        model = self.bci_trainer.train(
            states,
            actions,
            next_states,
            signals,
            run_id=f"{self.run_id}_bci",
            budget=self.budget,
        )
        self.model = model
        return model

__all__ = ["EdgeRLTrainer"]
