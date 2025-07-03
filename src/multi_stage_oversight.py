from __future__ import annotations

from typing import Iterable, Sequence, Callable, List, Tuple

import torch

from .collective_constitution import CollectiveConstitution
from .deliberative_alignment import DeliberativeAligner
from .critic_rlhf import CriticScorer, CriticRLHFTrainer
from .formal_verifier import verify_model, VerificationResult


class MultiStageOversight:
    """Combine multiple safety checks for model outputs."""

    def __init__(
        self,
        principles: Iterable[str],
        policy_text: str,
        actions: Sequence[str],
        checks: Iterable[Callable[[], Tuple[bool, str]]] | None = None,
    ) -> None:
        self.constitution = CollectiveConstitution()
        self.rules = self.constitution.derive_rules(principles)
        self.aligner = DeliberativeAligner(policy_text)
        self.scorer = CriticScorer()
        self.trainer = CriticRLHFTrainer(torch.nn.Linear(1, len(actions)), actions, self.scorer)
        self.checks = list(checks or [])

    # --------------------------------------------------------------
    def review(self, text: str) -> tuple[bool, List[str]]:
        """Run all oversight stages and return ``(passed, messages)``."""
        messages: List[str] = []

        # Constitutional filter
        if not self.constitution.label_responses([text], self.rules)[0][1]:
            messages.append("constitution_fail")

        # Deliberative alignment
        if not self.aligner.analyze(text):
            messages.append("alignment_fail")

        # Critic RLHF training step
        _ = self.trainer.train_step(torch.zeros(1, 1))
        if self.scorer.score(text) < 0:
            messages.append("critic_flag")

        # Formal verification of the policy network
        if self.checks:
            result: VerificationResult = verify_model(self.trainer.model, self.checks)
            if not result.passed:
                messages.extend(result.messages)

        return len(messages) == 0, messages


__all__ = ["MultiStageOversight"]
