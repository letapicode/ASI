import random
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
    """Hyper-parameter search using simulated amplitude estimation."""

    def __init__(self, eval_func: Callable[[Any], bool], param_space: Iterable[Any]):
        self.eval_func = eval_func
        self.param_space = list(param_space)

    def search(
        self, num_samples: int = 10, shots: int = 32, method: str = "standard"
    ) -> Tuple[Any, float]:
        """Evaluate a subset of parameters and return the best one.

        Args:
            num_samples: Number of parameter settings to evaluate.
            shots: Number of oracle calls per setting.
            method: ``"standard"`` uses ``amplitude_estimate`` while
                ``"bayesian"`` applies ``amplitude_estimate_bayesian``.
        Returns:
            Tuple of (best_param, estimated_probability).
        """
        candidates = random.sample(self.param_space, min(num_samples, len(self.param_space)))
        best_param = None
        best_prob = -1.0
        for param in candidates:
            if method == "bayesian":
                prob = amplitude_estimate_bayesian(
                    lambda: self.eval_func(param), shots
                )
            elif method == "standard":
                prob = amplitude_estimate(lambda: self.eval_func(param), shots)
            else:
                raise ValueError(f"Unknown method: {method}")
            if prob > best_prob:
                best_prob = prob
                best_param = param
        return best_param, best_prob
