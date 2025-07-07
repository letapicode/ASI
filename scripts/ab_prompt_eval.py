from __future__ import annotations

import argparse

from asi.ab_evaluator import run_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare two prompt optimizer variants"
    )
    parser.add_argument("--config-a", required=True, help="Baseline config")
    parser.add_argument("--config-b", required=True, help="New config")
    args = parser.parse_args(argv)

    res_a = run_config(args.config_a)
    res_b = run_config(args.config_b)

    eng_a = sum(int(r[0]) for r in res_a.values())
    eng_b = sum(int(r[0]) for r in res_b.values())
    print(f"Engagement A: {eng_a} B: {eng_b} delta={eng_b - eng_a}")


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
