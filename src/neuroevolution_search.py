"""Evolutionary neural architecture search."""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, Iterable, Tuple, List

from .telemetry import TelemetryLogger


class NeuroevolutionSearch:
    """Simple neuroevolution over discrete architecture parameters."""

    def __init__(
        self,
        search_space: Dict[str, Iterable[Any]],
        eval_func: Callable[[Dict[str, Any]], float],
        population_size: int = 8,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.5,
        *,
        telemetry: TelemetryLogger | None = None,
        energy_weight: float = 0.0,
    ) -> None:
        if population_size < 2:
            raise ValueError("population_size must be >=2")
        if not (0.0 <= mutation_rate <= 1.0):
            raise ValueError("mutation_rate must be between 0 and 1")
        if not (0.0 <= crossover_rate <= 1.0):
            raise ValueError("crossover_rate must be between 0 and 1")
        self.search_space = {k: list(v) for k, v in search_space.items()}
        self.eval_func = eval_func
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.telemetry = telemetry or TelemetryLogger(interval=0.1)
        self.energy_weight = energy_weight
        self.history: List[float] = []

    # --------------------------------------------------------------
    def sample(self) -> Dict[str, Any]:
        return {k: random.choice(v) for k, v in self.search_space.items()}

    def _mutate(self, indiv: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(indiv)
        for k, choices in self.search_space.items():
            if random.random() < self.mutation_rate:
                out[k] = random.choice(choices)
        return out

    def _crossover(self, a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        child = {}
        for k in self.search_space.keys():
            if random.random() < self.crossover_rate:
                child[k] = a[k]
            else:
                child[k] = b[k]
        return child

    # --------------------------------------------------------------
    def search(self, generations: int = 5) -> Tuple[Dict[str, Any], float]:
        if generations <= 0:
            raise ValueError("generations must be positive")
        population = [self.sample() for _ in range(self.population_size)]
        best_cfg: Dict[str, Any] | None = None
        best_score = float("-inf")
        self.history.clear()
        self.telemetry.start()
        prev_energy = self.telemetry.get_stats().get("energy_kwh", 0.0)
        for _ in range(generations):
            scored = []
            for indiv in population:
                pre = self.telemetry.get_stats().get("energy_kwh", 0.0)
                score = float(self.eval_func(indiv))
                post = self.telemetry.get_stats().get("energy_kwh", 0.0)
                energy = post - pre
                prev_energy = post
                adj = score - self.energy_weight * energy
                scored.append((indiv, score, adj))
            scored.sort(key=lambda x: x[2], reverse=True)
            self.history.append(scored[0][1])
            if scored[0][1] > best_score:
                best_score = scored[0][1]
                best_cfg = scored[0][0]
            survivors = [cfg for cfg, _, _ in scored[: max(1, self.population_size // 2)]]
            new_pop = survivors[:]
            while len(new_pop) < self.population_size:
                p1, p2 = random.sample(survivors, 2)
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                new_pop.append(child)
            population = new_pop
        self.telemetry.stop()
        assert best_cfg is not None
        return best_cfg, best_score


__all__ = ["NeuroevolutionSearch"]
