import json
from urllib.request import Request, urlopen
from typing import List, Dict, Any, Optional

import asyncio
import aiohttp


def _github_api(path: str, token: str | None = None) -> Any:
    """Call the GitHub API synchronously and return parsed JSON."""
    url = f"https://api.github.com/{path}"
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(url, headers=headers)
    with urlopen(req) as resp:
        return json.loads(resp.read().decode())


async def _github_api_async(path: str, token: str | None = None, session: aiohttp.ClientSession | None = None) -> Any:
    """Asynchronously call the GitHub API and return parsed JSON."""
    url = f"https://api.github.com/{path}"
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if session is None:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as resp:
                return await resp.json()
    async with session.get(url, headers=headers) as resp:
        return await resp.json()


def list_open_prs(repo: str, token: str | None = None) -> List[Dict[str, Any]]:
    """Return a list of open pull requests for ``repo``.

    Each element contains ``number`` and ``title`` keys."""
    data = _github_api(f"repos/{repo}/pulls?state=open", token)
    return [{"number": pr["number"], "title": pr["title"]} for pr in data]


async def list_open_prs_async(repo: str, token: str | None = None) -> List[Dict[str, Any]]:
    """Asynchronously return a list of open pull requests for ``repo``."""
    data = await _github_api_async(f"repos/{repo}/pulls?state=open", token)
    return [{"number": pr["number"], "title": pr["title"]} for pr in data]


def check_mergeable(repo: str, pr_number: int, token: str | None = None) -> Optional[bool]:
    """Return whether the pull request can be merged cleanly."""
    pr = _github_api(f"repos/{repo}/pulls/{pr_number}", token)
    return pr.get("mergeable")


async def check_mergeable_async(repo: str, pr_number: int, token: str | None = None) -> Optional[bool]:
    """Asynchronously return whether the pull request can be merged cleanly."""
    pr = await _github_api_async(f"repos/{repo}/pulls/{pr_number}", token)
    return pr.get("mergeable")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="List open PRs and mergeability")
    parser.add_argument("repo", help="<owner>/<repo> to query")
    parser.add_argument("--token", help="GitHub token", default=None)
    parser.add_argument("--use-asyncio", action="store_true", help="Use asyncio for concurrent calls")
    args = parser.parse_args()

    if args.use_asyncio:
        async def run() -> None:
            prs = await list_open_prs_async(args.repo, args.token)
            results = await asyncio.gather(*(check_mergeable_async(args.repo, pr["number"], args.token) for pr in prs))
            for pr, mergeable in zip(prs, results):
                print(f"#{pr['number']} {pr['title']} - mergeable: {mergeable}")
        asyncio.run(run())
    else:
        prs = list_open_prs(args.repo, args.token)
        for pr in prs:
            mergeable = check_mergeable(args.repo, pr["number"], args.token)
            print(f"#{pr['number']} {pr['title']} - mergeable: {mergeable}")


if __name__ == "__main__":
    main()
