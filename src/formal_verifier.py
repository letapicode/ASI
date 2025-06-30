from typing import Iterable, Callable

import torch


def verify_model(
    model: torch.nn.Module,
    invariants: Iterable[Callable[[torch.nn.Module], bool]],
) -> bool:
    """Return ``True`` if all ``invariants`` hold for ``model``."""
    for check in invariants:
        if not check(model):
            return False
    return True


def weight_bound(bound: float) -> Callable[[torch.nn.Module], bool]:
    def predicate(model: torch.nn.Module) -> bool:
        return all(p.abs().max() <= bound for p in model.parameters())

    return predicate


