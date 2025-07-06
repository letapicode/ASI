"""A/B evaluation wrapper around :mod:`eval_harness`."""

from __future__ import annotations

import argparse
import json
import asyncio
from pathlib import Path
from typing import Any, Dict, Tuple

from .eval_harness import parse_modules, evaluate_modules, evaluate_modules_async

ResultDict = Dict[str, Tuple[bool, str]]


def _load_config(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def _modules_from_cfg(cfg: Dict[str, Any]) -> list[str]:
    if "modules" in cfg:
        return list(cfg["modules"])
    plan = cfg.get("plan", "docs/Plan.md")
    return parse_modules(plan)


def run_config(path: str | Path, *, concurrent: bool = False) -> ResultDict:
    cfg = _load_config(path)
    mods = _modules_from_cfg(cfg)
    if concurrent:
        return asyncio.run(evaluate_modules_async(mods))
    return evaluate_modules(mods)


def compare_results(a: ResultDict, b: ResultDict) -> str:
    modules = sorted(set(a) | set(b))
    lines = []
    pass_a = sum(int(res[0]) for res in a.values())
    pass_b = sum(int(res[0]) for res in b.values())
    lines.append(
        f"Pass rate A: {pass_a}/{len(a)} B: {pass_b}/{len(b)} delta={pass_b-pass_a}"
    )
    for m in modules:
        ok_a = int(a.get(m, (False, ""))[0])
        ok_b = int(b.get(m, (False, ""))[0])
        delta = ok_b - ok_a
        lines.append(f"{m}: {ok_a}->{ok_b} delta={delta}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare two eval harness runs")
    parser.add_argument("--config-a", required=True, help="Baseline config JSON")
    parser.add_argument("--config-b", required=True, help="New config JSON")
    parser.add_argument("--concurrent", action="store_true", help="Use asyncio")
    args = parser.parse_args(argv)

    res_a = run_config(args.config_a, concurrent=args.concurrent)
    res_b = run_config(args.config_b, concurrent=args.concurrent)
    print(compare_results(res_a, res_b))


if __name__ == "__main__":  # pragma: no cover
    main()
