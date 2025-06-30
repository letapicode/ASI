from __future__ import annotations

import subprocess
from typing import Any, Dict, List

from .pull_request_monitor import list_open_prs


def _check_conflict(branch_ref: str, base: str = "HEAD") -> bool:
    """Return True if merging ``branch_ref`` into ``base`` would conflict."""
    subprocess.run(["git", "fetch", "origin", branch_ref], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    base_commit = subprocess.check_output(["git", "merge-base", base, "FETCH_HEAD"]).decode().strip()
    proc = subprocess.run(
        ["git", "merge-tree", base_commit, base, "FETCH_HEAD"],
        capture_output=True,
        text=True,
    )
    return "<<<<<<<" in proc.stdout


def list_conflicts(repo: str, token: str | None = None) -> List[Dict[str, Any]]:
    """List open PRs for ``repo`` and whether they have merge conflicts."""
    prs = list_open_prs(repo, token)
    results = []
    for pr in prs:
        ref = f"pull/{pr['number']}/head"
        conflicts = _check_conflict(ref)
        results.append({"number": pr["number"], "title": pr["title"], "conflicts": conflicts})
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Check PRs for merge conflicts")
    parser.add_argument("repo", help="<owner>/<repo> to query")
    parser.add_argument("--token", default=None, help="GitHub token")
    args = parser.parse_args()

    for info in list_conflicts(args.repo, args.token):
        status = "conflicts" if info["conflicts"] else "clean"
        print(f"#{info['number']} {info['title']} - {status}")


if __name__ == "__main__":
    main()

__all__ = ["list_conflicts"]
