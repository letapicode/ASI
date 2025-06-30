import subprocess
from typing import Dict

from .autobench import BenchResult, summarize_results
from .pull_request_monitor import list_open_prs


def check_pr_conflicts(repo: str, token: str | None = None, remote: str = "origin") -> Dict[str, BenchResult]:
    """Return merge-conflict status for all open PRs."""
    results: Dict[str, BenchResult] = {}
    prs = list_open_prs(repo, token)
    for pr in prs:
        pr_ref = f"refs/pull/{pr['number']}/head"
        local_ref = f"pr/{pr['number']}"
        fetch = subprocess.run(
            ["git", "fetch", remote, f"{pr_ref}:{local_ref}"],
            capture_output=True,
            text=True,
        )
        key = f"#{pr['number']} {pr['title']}"
        if fetch.returncode != 0:
            results[key] = BenchResult(False, fetch.stdout + fetch.stderr)
            continue
        base = subprocess.check_output(
            ["git", "merge-base", f"{remote}/main", local_ref],
            text=True,
        ).strip()
        merge = subprocess.run(
            ["git", "merge-tree", base, f"{remote}/main", local_ref],
            capture_output=True,
            text=True,
        )
        conflict = "<<<<<<<" in merge.stdout
        results[key] = BenchResult(not conflict, merge.stdout + merge.stderr)
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Check merge conflicts for open PRs")
    parser.add_argument("repo", help="<owner>/<repo>")
    parser.add_argument("--token", default=None, help="GitHub token")
    parser.add_argument("--remote", default="origin", help="Git remote to use")
    args = parser.parse_args()

    results = check_pr_conflicts(args.repo, args.token, args.remote)
    print(summarize_results(results))


if __name__ == "__main__":
    main()
