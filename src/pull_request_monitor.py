import json
from urllib.request import Request, urlopen
from typing import List, Dict, Any, Optional


def _github_api(path: str, token: str | None = None) -> Any:
    """Call the GitHub API and return parsed JSON."""
    url = f"https://api.github.com/{path}"
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(url, headers=headers)
    with urlopen(req) as resp:
        return json.loads(resp.read().decode())


def list_open_prs(repo: str, token: str | None = None) -> List[Dict[str, Any]]:
    """Return a list of open pull requests for ``repo``.

    Each element contains ``number`` and ``title`` keys."""
    data = _github_api(f"repos/{repo}/pulls?state=open", token)
    return [{"number": pr["number"], "title": pr["title"]} for pr in data]


def check_mergeable(repo: str, pr_number: int, token: str | None = None) -> Optional[bool]:
    """Return whether the pull request can be merged cleanly."""
    pr = _github_api(f"repos/{repo}/pulls/{pr_number}", token)
    return pr.get("mergeable")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="List open PRs and mergeability")
    parser.add_argument("repo", help="<owner>/<repo> to query")
    parser.add_argument("--token", help="GitHub token", default=None)
    args = parser.parse_args()

    prs = list_open_prs(args.repo, args.token)
    for pr in prs:
        mergeable = check_mergeable(args.repo, pr["number"], args.token)
        print(f"#{pr['number']} {pr['title']} - mergeable: {mergeable}")


if __name__ == "__main__":
    main()
