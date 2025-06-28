import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass
class BenchResult:
    """Result of running a single test module."""

    passed: bool
    output: str


def _run_test_file(path: Path) -> BenchResult:
    """Execute a test file in a subprocess and collect its result."""
    proc = subprocess.run(
        [sys.executable, str(path)],
        capture_output=True,
        text=True,
    )
    return BenchResult(passed=proc.returncode == 0, output=proc.stdout + proc.stderr)


def run_autobench(test_dir: str = "tests") -> Dict[str, BenchResult]:
    """Run each test file individually and return their results."""
    results: Dict[str, BenchResult] = {}
    for test_file in sorted(Path(test_dir).glob("test_*.py")):
        results[test_file.name] = _run_test_file(test_file)
    return results


def summarize_results(results: Dict[str, BenchResult]) -> str:
    """Return a formatted summary scoreboard."""
    total = len(results)
    passed = sum(1 for r in results.values() if r.passed)
    lines = [f"Passed {passed}/{total} modules"]
    for name, res in sorted(results.items()):
        status = "PASS" if res.passed else "FAIL"
        lines.append(f"{name}: {status}")
        if not res.passed and res.output:
            snippet = res.output.strip()
            if snippet:
                lines.append(snippet)
    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run tests in isolation")
    parser.add_argument(
        "--test-dir", default="tests", help="Directory containing test files"
    )
    args = parser.parse_args()

    results = run_autobench(args.test_dir)
    print(summarize_results(results))


if __name__ == "__main__":
    main()
