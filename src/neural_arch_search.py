"""Distributed neural architecture search via multiprocessing."""

from __future__ import annotations

import random
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, Iterable, Tuple, List

from .neuroevolution_search import NeuroevolutionSearch

from .telemetry import TelemetryLogger


class DistributedArchSearch:
    """Explore architecture configurations using multiple worker processes."""

    def __init__(
        self,
        search_space: Dict[str, Iterable[Any]],
        eval_func: Callable[[Dict[str, Any]], float],
        max_workers: int = 2,
        *,
        telemetry: TelemetryLogger | None = None,
        energy_weight: float = 0.0,
    ) -> None:
        self.search_space = {k: list(v) for k, v in search_space.items()}
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self.eval_func = eval_func
        self.max_workers = max_workers
        self.telemetry = telemetry or TelemetryLogger(interval=0.1)
        self.energy_weight = energy_weight

    def sample(self) -> Dict[str, Any]:
        """Randomly draw a configuration from the search space."""
        return {k: random.choice(v) for k, v in self.search_space.items()}

    def _all_configs(self) -> List[Dict[str, Any]]:
        keys = list(self.search_space.keys())
        combos = product(*(self.search_space[k] for k in keys))
        return [dict(zip(keys, vals)) for vals in combos]

    def search(self, num_samples: int = 10) -> Tuple[Dict[str, Any], float]:
        """Evaluate up to ``num_samples`` configs and return the best one."""
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")

        all_cfgs = self._all_configs()
        candidates = random.sample(all_cfgs, min(num_samples, len(all_cfgs)))

        best_cfg: Dict[str, Any] | None = None
        best_adj = float("-inf")
        best_score = float("-inf")

        self.telemetry.start()
        prev_energy = self.telemetry.get_stats().get("energy_kwh", 0.0)

        def eval_candidate(cfg: Dict[str, Any]) -> Tuple[float, float]:
            nonlocal prev_energy
            pre = self.telemetry.get_stats().get("energy_kwh", 0.0)
            score = float(self.eval_func(cfg))
            post = self.telemetry.get_stats().get("energy_kwh", 0.0)
            energy = post - pre
            prev_energy = post
            adj = score - self.energy_weight * energy
            return score, adj

        if self.max_workers == 1:
            for cfg in candidates:
                score, adj = eval_candidate(cfg)
                if adj > best_adj:
                    best_adj = adj
                    best_score = score
                    best_cfg = cfg
        else:
            with ProcessPoolExecutor(max_workers=self.max_workers) as ex:
                fut_to_cfg = {ex.submit(self.eval_func, cfg): cfg for cfg in candidates}
                for fut in as_completed(fut_to_cfg):
                    cfg = fut_to_cfg[fut]
                    score = float(fut.result())
                    now = self.telemetry.get_stats().get("energy_kwh", 0.0)
                    energy = now - prev_energy
                    prev_energy = now
                    adj = score - self.energy_weight * energy
                    if adj > best_adj:
                        best_adj = adj
                        best_score = score
                        best_cfg = cfg

        self.telemetry.stop()
        assert best_cfg is not None
        return best_cfg, best_score


def search_architecture(
    search_space: Dict[str, Iterable[Any]],
    eval_func: Callable[[Dict[str, Any]], float],
    *,
    method: str = "gradient",
    max_workers: int = 2,
    generations: int = 5,
    num_samples: int = 10,
    population_size: int = 8,
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.5,
    telemetry: TelemetryLogger | None = None,
    energy_weight: float = 0.0,
) -> Tuple[Dict[str, Any], float]:
    """Unified API for gradient or evolutionary architecture search."""
    if method == "evolution":
        search = NeuroevolutionSearch(
            search_space,
            eval_func,
            population_size=population_size,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            telemetry=telemetry,
            energy_weight=energy_weight,
        )
        return search.search(generations=generations)
    if method == "gradient":
        search = DistributedArchSearch(
            search_space,
            eval_func,
            max_workers=max_workers,
            telemetry=telemetry,
            energy_weight=energy_weight,
        )
        return search.search(num_samples=num_samples)
    raise ValueError(f"Unknown method: {method}")


__all__ = ["DistributedArchSearch", "search_architecture", "NeuroevolutionSearch"]

