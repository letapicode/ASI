"""Distributed neural architecture search via multiprocessing."""

from __future__ import annotations

import random
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, Iterable, Tuple, List


class DistributedArchSearch:
    """Explore architecture configurations using multiple worker processes."""

    def __init__(
        self,
        search_space: Dict[str, Iterable[Any]],
        eval_func: Callable[[Dict[str, Any]], float],
        max_workers: int = 2,
    ) -> None:
        self.search_space = {k: list(v) for k, v in search_space.items()}
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self.eval_func = eval_func
        self.max_workers = max_workers

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
        best_score = float("-inf")
        if self.max_workers == 1:
            for cfg in candidates:
                score = float(self.eval_func(cfg))
                if score > best_score:
                    best_score = score
                    best_cfg = cfg
        else:
            with ProcessPoolExecutor(max_workers=self.max_workers) as ex:
                fut_to_cfg = {ex.submit(self.eval_func, cfg): cfg for cfg in candidates}
                for fut in as_completed(fut_to_cfg):
                    cfg = fut_to_cfg[fut]
                    score = float(fut.result())
                    if score > best_score:
                        best_score = score
                        best_cfg = cfg
        assert best_cfg is not None
        return best_cfg, best_score


__all__ = ["DistributedArchSearch"]
