import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict
from urllib.request import Request, urlopen


def list_open_prs(repo: str, token: str | None = None) -> List[Dict[str, str]]:
    url = f"https://api.github.com/repos/{repo}/pulls?state=open"
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(url, headers=headers)
    with urlopen(req) as resp:
        data = json.loads(resp.read().decode())
    return [{"number": pr["number"], "title": pr["title"]} for pr in data]


def pr_has_conflict(pr_number: int, remote: str = "origin") -> bool:
    subprocess.run([
        "git",
        "fetch",
        remote,
        f"pull/{pr_number}/head:pr/{pr_number}",
    ], check=True)
    merge_base = subprocess.check_output([
        "git",
        "merge-base",
        "main",
        f"pr/{pr_number}",
    ]).decode().strip()
    out = subprocess.check_output([
        "git",
        "merge-tree",
        merge_base,
        "main",
        f"pr/{pr_number}",
    ]).decode()
    return "<<<<<<<" in out or ">>>>>>>" in out


def summarize(repo: str, token: str | None = None, remote: str = "origin") -> str:
    prs = list_open_prs(repo, token)
    statuses = {pr["number"]: pr_has_conflict(pr["number"], remote) for pr in prs}
    total = len(prs)
    conflict_count = sum(1 for c in statuses.values() if c)
    lines = [f"Conflicts in {conflict_count}/{total} PRs"]
    for pr in prs:
        status = "CONFLICT" if statuses[pr["number"]] else "NO CONFLICT"
        lines.append(f"#{pr['number']} {pr['title']}: {status}")
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Summarize merge conflicts for open PRs")
    parser.add_argument("repo", help="<owner>/<repo> to query")
    parser.add_argument("--token", default=None, help="GitHub token")
    parser.add_argument("--remote", default="origin", help="Git remote")
    args = parser.parse_args(argv)

    print(summarize(args.repo, args.token, args.remote))


if __name__ == "__main__":
    main()
