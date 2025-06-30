import argparse
import subprocess
from pathlib import Path
from typing import Dict

from .autobench import BenchResult, summarize_results
from .pull_request_monitor import list_open_prs


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
    return ("<<<<<<<" in proc.stdout), proc.stdout


def check_pr_conflicts(
    repo: str,
    token: str | None = None,
    base: str = "main",
    remote: str = "origin",
    repo_path: str | Path = ".",
) -> Dict[str, BenchResult]:
    cwd = Path(repo_path)
    prs = list_open_prs(repo, token)
    results: Dict[str, BenchResult] = {}
    for pr in prs:
        branch_name = f"pr_{pr['number']}"
        fetch = subprocess.run(
            ["git", "fetch", remote, f"refs/pull/{pr['number']}/head:{branch_name}"],
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        key = f"PR {pr['number']}"
        if fetch.returncode != 0:
            results[key] = BenchResult(False, fetch.stdout + fetch.stderr)
            continue
        conflict, out = _check_branch_conflict(base, branch_name, cwd)
        results[key] = BenchResult(not conflict, out)
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Check merge conflicts for open PRs")
    parser.add_argument("repo", help="<owner>/<repo>")
    parser.add_argument("--token", default=None, help="GitHub token")
    parser.add_argument("--base", default="main", help="Base branch")
    parser.add_argument("--remote", default="origin", help="Remote name")
    parser.add_argument("--repo-path", default=".", help="Local repository path")
    args = parser.parse_args(argv)

    results = check_pr_conflicts(
        args.repo,
        token=args.token,
        base=args.base,
        remote=args.remote,
        repo_path=args.repo_path,
    )
    print(summarize_results(results))


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
