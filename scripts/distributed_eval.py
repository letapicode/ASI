"""Run eval_harness across multiple processes or remote hosts."""

from __future__ import annotations

import argparse
import json
import subprocess
from multiprocessing import get_context
from typing import Dict, Iterable, List, Tuple

from asi.eval_harness import parse_modules, evaluate_modules, format_results


def _chunk(lst: List[str], n: int) -> List[List[str]]:
    """Split ``lst`` into ``n`` roughly equal chunks."""
    n = max(1, n)
    return [lst[i::n] for i in range(n)]


def _run_local(mods: Iterable[str]) -> Dict[str, Tuple[bool, str]]:
    return evaluate_modules(mods)


def _run_remote(host: str, mods: Iterable[str]) -> Dict[str, Tuple[bool, str]]:
    script = (
        "import json,sys; from asi.eval_harness import evaluate_modules;"
        "mods=json.loads(sys.argv[1]);"
        "print(json.dumps(evaluate_modules(mods)))"
    )
    cmd = ["ssh", host, "python", "-c", script, json.dumps(list(mods))]
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)


def _aggregate(results: Iterable[Dict[str, Tuple[bool, str]]]) -> Dict[str, Tuple[bool, str]]:
    merged: Dict[str, Tuple[bool, str]] = {}
    for res in results:
        merged.update(res)
    return merged


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Distributed evaluation runner")
    parser.add_argument("--plan", default="docs/Plan.md", help="Plan listing modules")
    parser.add_argument("--workers", type=int, default=1, help="Local worker processes")
    parser.add_argument("--hosts", nargs="*", help="Remote hosts accessible via ssh")
    args = parser.parse_args(argv)

    modules = parse_modules(args.plan)

    results: List[Dict[str, Tuple[bool, str]]]
    if args.hosts:
        parts = _chunk(modules, len(args.hosts))
        results = [_run_remote(h, m) for h, m in zip(args.hosts, parts)]
    else:
        parts = _chunk(modules, args.workers)
        ctx = get_context("spawn")
        with ctx.Pool(len(parts)) as pool:
            results = pool.map(_run_local, parts)

    merged = _aggregate(results)
    print(format_results(merged))


if __name__ == "__main__":  # pragma: no cover - entry point
    main()

