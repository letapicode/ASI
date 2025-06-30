import subprocess
from typing import Dict, List

from pull_request_monitor import list_open_prs


def fetch_pr(remote: str, pr_number: int) -> None:
    """Fetch the pull request head locally."""
    subprocess.run(
        ["git", "fetch", remote, f"pull/{pr_number}/head:pr/{pr_number}"],
        check=True,
        capture_output=True,
    )


def pr_has_conflict(pr_number: int, remote: str = "origin") -> bool:
    """Return True if merging the PR into main would conflict."""
    fetch_pr(remote, pr_number)
    merge_base = (
        subprocess.check_output(["git", "merge-base", "main", f"pr/{pr_number}"])
        .decode()
        .strip()
    )
    proc = subprocess.run(
        ["git", "merge-tree", merge_base, "main", f"pr/{pr_number}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return "<<<<<<<" in proc.stdout or ">>>>>>>" in proc.stdout


def check_all_prs(repo: str, token: str | None = None, remote: str = "origin") -> Dict[int, bool]:
    """Return a mapping of PR number to conflict status."""
    prs = list_open_prs(repo, token)
    return {pr["number"]: pr_has_conflict(pr["number"], remote) for pr in prs}


def summarize_conflicts(prs: List[Dict[str, str]], conflicts: Dict[int, bool]) -> str:
    """Return formatted scoreboard of conflict status."""
    total = len(prs)
    conflict_count = sum(1 for c in conflicts.values() if c)
    lines = [f"Conflicts in {conflict_count}/{total} PRs"]
    for pr in prs:
        status = "CONFLICT" if conflicts[pr["number"]] else "NO CONFLICT"
        lines.append(f"#{pr['number']} {pr['title']}: {status}")
    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Check open PRs for merge conflicts")
    parser.add_argument("repo", help="<owner>/<repo> to query")
    parser.add_argument("--token", default=None, help="GitHub token")
    parser.add_argument("--remote", default="origin", help="Git remote to fetch PRs from")
    args = parser.parse_args()

    prs = list_open_prs(args.repo, args.token)
    conflicts = {pr["number"]: pr_has_conflict(pr["number"], args.remote) for pr in prs}
    print(summarize_conflicts(prs, conflicts))


if __name__ == "__main__":
    main()
