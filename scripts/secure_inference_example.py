#!/usr/bin/env python
"""Demonstrate running inference in a simulated enclave."""

from __future__ import annotations

import torch
from asi.enclave_runner import EnclaveRunner


def model(x: torch.Tensor) -> torch.Tensor:
    return x + 1


def main() -> None:
    runner = EnclaveRunner()
    inp = torch.tensor([3.0, 4.0])
    out = runner.run(model, inp)
    print("output", out)


if __name__ == "__main__":  # pragma: no cover - demo
    main()
