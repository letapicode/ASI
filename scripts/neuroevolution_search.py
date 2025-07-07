import argparse
import json
from pathlib import Path
from typing import Any, Dict

from asi.neuroevolution_search import NeuroevolutionSearch
from asi.eval_harness import evaluate_modules


def _load_space(path: str | Path) -> Dict[str, list[Any]]:
    data = json.loads(Path(path).read_text())
    return {k: list(v) for k, v in data.items()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run neuroevolution search")
    parser.add_argument("--space", required=True, help="JSON file describing search space")
    parser.add_argument("--generations", type=int, default=3, help="Number of generations")
    parser.add_argument("--population", type=int, default=4, help="Population size")
    parser.add_argument("--modules", nargs="*", help="Modules to benchmark")
    args = parser.parse_args(argv)

    space = _load_space(args.space)

    def eval_cfg(cfg: Dict[str, Any]) -> float:
        modules = args.modules or cfg.get("modules", [])
        if not modules:
            return 0.0
        res = evaluate_modules(modules)
        return sum(1.0 for ok, _ in res.values() if ok) / len(modules)

    search = NeuroevolutionSearch(
        space,
        eval_cfg,
        population_size=args.population,
        mutation_rate=0.3,
        crossover_rate=0.5,
    )
    best, score = search.evolve(generations=args.generations)
    print(json.dumps({"best": best, "score": score}, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
