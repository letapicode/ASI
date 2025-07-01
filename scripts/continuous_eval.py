import argparse
from asi.eval_harness import parse_modules, evaluate_modules, format_results
from asi.autobench import run_autobench, summarize_results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run eval_harness and autobench for continuous evaluation"
    )
    parser.add_argument(
        "--plan", default="docs/Plan.md", help="Path to Plan.md listing modules"
    )
    parser.add_argument(
        "--test-dir", default="tests", help="Directory containing test files"
    )
    args = parser.parse_args(argv)

    print("=== Eval Harness ===")
    modules = parse_modules(args.plan)
    results = evaluate_modules(modules)
    print(format_results(results))

    print("\n=== Autobench ===")
    bench = run_autobench(args.test_dir)
    print(summarize_results(bench))


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
