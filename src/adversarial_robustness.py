from __future__ import annotations

from typing import Callable, List, Sequence, Tuple
import threading
import time


class AdversarialRobustnessSuite:
    """Generate simple adversarial prompts for evaluation."""

    def __init__(self, model: Callable[[str], float]) -> None:
        self.model = model

    def generate(self, prompt: str, candidates: List[str]) -> str:
        """Return the most adversarial variant from ``candidates``."""
        base = self.model(prompt)
        worst = prompt
        worst_score = base
        for c in candidates:
            s = self.model(c)
            if s < worst_score:
                worst_score = s
                worst = c
        return worst


class AdversarialRobustnessScheduler:
    """Periodically evaluate adversarial prompts."""

    def __init__(
        self,
        model: Callable[[str], float],
        prompts: Sequence[Tuple[str, List[str]]],
        interval: float = 3600.0,
        callback: Callable[[float], None] | None = None,
    ) -> None:
        self.suite = AdversarialRobustnessSuite(model)
        self.prompts = list(prompts)
        self.interval = interval
        self.callback = callback or (lambda _score: None)
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    # --------------------------------------------------------------
    def start(self) -> None:
        if not self.thread.is_alive():
            self._stop.clear()
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

    # --------------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            score = self.run_once()
            self.callback(score)
            time.sleep(self.interval)

    # --------------------------------------------------------------
    def run_once(self) -> float:
        scores = []
        for prompt, candidates in self.prompts:
            adv = self.suite.generate(prompt, list(candidates))
            score = self.suite.model(adv)
            scores.append(score)
        return sum(scores) / len(scores) if scores else 0.0

__all__ = ["AdversarialRobustnessSuite", "AdversarialRobustnessScheduler"]
