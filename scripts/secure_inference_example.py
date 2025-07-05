#!/usr/bin/env python
"""Demonstrate running inference in a simulated enclave."""

from __future__ import annotations

import torch
from asi.enclave_runner import EnclaveRunner
from asi.fhe_runner import run_fhe

import tenseal as ts
import argparse


def model(x: torch.Tensor) -> torch.Tensor:
    return x + 1


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Enclave inference demo")
    parser.add_argument("--fhe", action="store_true", help="run model under FHE")
    args = parser.parse_args(argv)

    runner = EnclaveRunner()
    inp = torch.tensor([3.0, 4.0])
    if args.fhe:
        ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
        )
        ctx.generate_galois_keys()
        ctx.global_scale = 2**40
        out = runner.run(run_fhe, model, inp, ctx)
    else:
        out = runner.run(model, inp)
    print("output", out)


if __name__ == "__main__":  # pragma: no cover - demo
    main()
