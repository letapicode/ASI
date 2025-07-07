"""Population-based neuroevolution for model configurations."""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, Iterable, Tuple, List


class NeuroevolutionSearch:
    """Evolve model configs via mutation and crossover."""

    def __init__(
        self,
        search_space: Dict[str, Iterable[Any]],
        eval_func: Callable[[Dict[str, Any]], float],
        *,
        population_size: int = 4,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.5,
    ) -> None:
        self.search_space = {k: list(v) for k, v in search_space.items()}
        if population_size < 2:
            raise ValueError("population_size must be at least 2")
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be between 0 and 1")
        if not 0.0 <= crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be between 0 and 1")
        self.eval_func = eval_func
        self.population_size = population_size
        self.mutation_rate = float(mutation_rate)
        self.crossover_rate = float(crossover_rate)

    # ------------------------------------------------------------------
    def _random_cfg(self) -> Dict[str, Any]:
        return {k: random.choice(v) for k, v in self.search_space.items()}

    def _mutate(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        new = dict(cfg)
        for k, choices in self.search_space.items():
            if random.random() < self.mutation_rate:
                new[k] = random.choice(choices)
        return new

    def _crossover(self, a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k: (a[k] if random.random() < 0.5 else b[k]) for k in self.search_space
        }

    # ------------------------------------------------------------------
    def evolve(self, generations: int = 5) -> Tuple[Dict[str, Any], float]:
        """Run ``generations`` of evolution and return the best config."""
        if generations <= 0:
            raise ValueError("generations must be positive")
        population = [self._random_cfg() for _ in range(self.population_size)]
        scores = [float(self.eval_func(c)) for c in population]
        best_idx = max(range(self.population_size), key=lambda i: scores[i])
        best_cfg = population[best_idx]
        best_score = scores[best_idx]

        for _ in range(generations):
            ranked = sorted(zip(population, scores), key=lambda x: x[1], reverse=True)
            parents = [cfg for cfg, _ in ranked[: max(2, self.population_size // 2)]]
            next_pop: List[Dict[str, Any]] = []
            while len(next_pop) < self.population_size:
                if random.random() < self.crossover_rate and len(parents) >= 2:
                    p1, p2 = random.sample(parents, 2)
                    child = self._crossover(p1, p2)
                else:
                    p = random.choice(parents)
                    child = self._mutate(p)
                next_pop.append(child)
            population = next_pop
            scores = [float(self.eval_func(c)) for c in population]
            idx = max(range(self.population_size), key=lambda i: scores[i])
            if scores[idx] > best_score:
                best_score = scores[idx]
                best_cfg = population[idx]
        return best_cfg, best_score


__all__ = ["NeuroevolutionSearch"]
