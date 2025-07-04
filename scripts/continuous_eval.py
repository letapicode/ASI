import argparse
from asi.eval_harness import parse_modules, evaluate_modules, format_results
from asi.autobench import run_autobench, summarize_results
from asi.adversarial_robustness import AdversarialRobustnessScheduler
import yaml
from pathlib import Path


def load_adv_config(path: str) -> tuple[list[tuple[str, list[str]]], float]:
    """Return prompts and interval from ``path``."""
    file = Path(path)
    if not file.exists():
        return [], 0.0
    data = yaml.safe_load(file.read_text()) or {}
    prompts = [
        (p.get("prompt", ""), list(p.get("candidates", [])))
        for p in data.get("prompts", [])
    ]
    interval = float(data.get("interval", 3600))
    return prompts, interval


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
    parser.add_argument(
        "--adv-cfg",
        default="configs/adversarial_eval.yaml",
        help="Adversarial evaluation config",
    )
    args = parser.parse_args(argv)

    print("=== Eval Harness ===")
    modules = parse_modules(args.plan)
    results = evaluate_modules(modules)
    print(format_results(results))

    print("\n=== Autobench ===")
    bench = run_autobench(args.test_dir)
    print(summarize_results(bench))

    print("\n=== Adversarial Robustness ===")
    prompts, interval = load_adv_config(args.adv_cfg)
    model = lambda s: float(len(s))
    scheduler = AdversarialRobustnessScheduler(model, prompts, interval)
    score = scheduler.run_once()
    print(f"avg_adv_score={score:.3f}")


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
