import random
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, Tuple, Any


def amplitude_estimate(oracle: Callable[[], bool], shots: int = 32) -> float:
    """Estimate success probability using repeated oracle calls.

    This function simulates quantum amplitude estimation by
    measuring the oracle ``shots`` times and returning the
    observed frequency of ``True`` outcomes.

    Raises:
        ValueError: If ``shots`` is not a positive integer.
    """
    if shots <= 0:
        raise ValueError("shots must be positive")
    successes = sum(1 for _ in range(shots) if oracle())
    return successes / shots


def amplitude_estimate_bayesian(
    oracle: Callable[[], bool], shots: int = 32, alpha: float = 0.5, beta: float = 0.5
) -> float:
    """Estimate success probability using a Beta prior.

    This follows the maximum-likelihood amplitude estimation approach by
    updating a Beta distribution with ``alpha`` and ``beta`` hyperparameters.

    Args:
        oracle: Boolean-returning evaluation function.
        shots: Number of oracle calls to sample.
        alpha: Prior successes in the Beta distribution.
        beta: Prior failures in the Beta distribution.

    Returns:
        Posterior mean estimate of success probability.

    Raises:
        ValueError: If ``shots`` or hyperparameters are not positive.
    """
    if shots <= 0:
        raise ValueError("shots must be positive")
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be positive")
    successes = sum(1 for _ in range(shots) if oracle())
    return (successes + alpha) / (shots + alpha + beta)


class QAEHyperparamSearch:
    """Hyper-parameter and architecture search using simulated amplitude estimation."""

    def __init__(
        self,
        eval_func: Callable[..., bool],
        param_space: Iterable[Any],
        arch_space: Iterable[Any] | None = None,
    ) -> None:
        self.eval_func = eval_func
        self.param_space = list(param_space)
        self.arch_space = list(arch_space) if arch_space is not None else [None]

    def search(
        self,
        num_samples: int = 10,
        shots: int = 32,
        method: str = "standard",
        early_stop: float | None = None,
        max_workers: int | None = None,
    ) -> Tuple[Any, float]:
        """Evaluate parameters and architectures and return the best one.

        Args:
            num_samples: Number of parameter settings to evaluate.
            shots: Number of oracle calls per setting.
            method: ``"standard"`` uses ``amplitude_estimate`` while
                ``"bayesian"`` applies ``amplitude_estimate_bayesian``.
            early_stop: Optional probability threshold for early stopping.
                Evaluation halts when an estimate meets or exceeds this value.
            max_workers: If set, evaluate parameters in parallel using a
                ``ThreadPoolExecutor`` with the given worker count.

        Returns:
            Tuple of ``(best_param, estimated_probability)`` or
            ``((best_arch, best_param), estimated_probability)`` if
            architecture search is enabled.

        Raises:
            ValueError: If ``num_samples`` or ``shots`` are non-positive, or
                ``early_stop`` is not between 0 and 1, or ``max_workers`` is
                not ``None`` and less than 1.
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if shots <= 0:
            raise ValueError("shots must be positive")
        if early_stop is not None and not (0.0 <= early_stop <= 1.0):
            raise ValueError("early_stop must be between 0 and 1")
        if max_workers is not None and max_workers < 1:
            raise ValueError("max_workers must be positive")

        combos = list(itertools.product(self.arch_space, self.param_space))
        candidates = random.sample(combos, min(num_samples, len(combos)))
        best_arch = None
        best_param = None
        best_prob = -1.0

        def evaluate(candidate: Tuple[Any, Any]) -> Tuple[Any, Any, float]:
            arch, param = candidate
            if method == "bayesian":
                prob = amplitude_estimate_bayesian(
                    lambda: self.eval_func(arch, param) if arch is not None else self.eval_func(param),
                    shots,
                )
            elif method == "standard":
                prob = amplitude_estimate(
                    lambda: self.eval_func(arch, param) if arch is not None else self.eval_func(param),
                    shots,
                )
            else:
                raise ValueError(f"Unknown method: {method}")
            return arch, param, prob

        if max_workers and max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(evaluate, c): c for c in candidates}
                for fut in as_completed(futures):
                    arch, param, prob = fut.result()
                    if prob > best_prob:
                        best_prob = prob
                        best_arch = arch
                        best_param = param
                    if early_stop is not None and prob >= early_stop:
                        for f in futures:
                            f.cancel()
                        break
        else:
            for candidate in candidates:
                arch, param, prob = evaluate(candidate)
                if prob > best_prob:
                    best_prob = prob
                    best_arch = arch
                    best_param = param
                if early_stop is not None and prob >= early_stop:
                    break

        if self.arch_space == [None]:
            return best_param, best_prob
        return (best_arch, best_param), best_prob
