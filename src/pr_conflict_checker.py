import subprocess
from typing import Dict
from pathlib import Path

from .pull_request_monitor import list_open_prs
from .autobench import BenchResult, summarize_results


def _run_git(args: list[str], cwd: Path) -> str:
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    proc.check_returncode()
    return proc.stdout.strip()


def _check_branch_conflict(base: str, branch: str, cwd: Path) -> tuple[bool, str]:
    base_rev = _run_git(["merge-base", base, branch], cwd)
    proc = subprocess.run(
        ["git", "merge-tree", base_rev, base, branch],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    output = proc.stdout
    return ("<<<<<<<" in output), output


def check_pr_conflicts(
    repo: str,
    token: str | None = None,
    base: str = "main",
    remote: str = "origin",
    repo_path: str | Path = ".",
) -> Dict[str, BenchResult]:
    """Fetch open PRs and check if they merge cleanly with ``base``."""
    cwd = Path(repo_path)
    prs = list_open_prs(repo, token)
    results: Dict[str, BenchResult] = {}
    for pr in prs:
        branch_name = f"pr_{pr['number']}"
        subprocess.run(
            ["git", "fetch", remote, f"refs/pull/{pr['number']}/head:{branch_name}"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        conflict, out = _check_branch_conflict(base, branch_name, cwd)
        results[f"PR {pr['number']}"] = BenchResult(not conflict, out)
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Check merge conflicts for open PRs")
    parser.add_argument("repo", help="<owner>/<repo> to query")
    parser.add_argument("--token", default=None, help="GitHub token")
    parser.add_argument("--base", default="main", help="Base branch to merge into")
    parser.add_argument("--remote", default="origin", help="Remote name")
    parser.add_argument(
        "--repo-path",
        default=".",
        help="Path to the local clone of the repository",
    )
    args = parser.parse_args()

    results = check_pr_conflicts(
        args.repo,
        token=args.token,
        base=args.base,
        remote=args.remote,
        repo_path=args.repo_path,
    )
    print(summarize_results(results))


if __name__ == "__main__":
    main()
