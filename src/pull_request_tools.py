import argparse
import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen

import aiohttp

from .autobench import BenchResult, summarize_results


def _github_api(path: str, token: str | None = None) -> Any:
    """Call the GitHub API synchronously and return parsed JSON."""
    url = f"https://api.github.com/{path}"
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(url, headers=headers)
    with urlopen(req) as resp:
        return json.loads(resp.read().decode())


async def _github_api_async(
    path: str,
    token: str | None = None,
    session: aiohttp.ClientSession | None = None,
) -> Any:
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


async def list_open_prs_async(
    repo: str,
    token: str | None = None,
    session: aiohttp.ClientSession | None = None,
) -> List[Dict[str, Any]]:
    """Asynchronously return a list of open pull requests for ``repo``."""
    data = await _github_api_async(
        f"repos/{repo}/pulls?state=open", token, session=session
    )
    return [{"number": pr["number"], "title": pr["title"]} for pr in data]


def check_mergeable(repo: str, pr_number: int, token: str | None = None) -> Optional[bool]:
    """Return whether the pull request can be merged cleanly."""
    pr = _github_api(f"repos/{repo}/pulls/{pr_number}", token)
    return pr.get("mergeable")


async def check_mergeable_async(
    repo: str,
    pr_number: int,
    token: str | None = None,
    session: aiohttp.ClientSession | None = None,
) -> Optional[bool]:
    """Asynchronously return whether the pull request can be merged cleanly."""
    pr = await _github_api_async(
        f"repos/{repo}/pulls/{pr_number}", token, session=session
    )
    return pr.get("mergeable")


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


def main() -> None:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description="Pull request utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    mon = sub.add_parser("monitor", help="List open PRs and mergeability")
    mon.add_argument("repo", help="<owner>/<repo> to query")
    mon.add_argument("--token", help="GitHub token", default=None)
    mon.add_argument("--use-asyncio", action="store_true", help="Use asyncio for concurrent calls")

    conf = sub.add_parser("conflicts", help="Check merge conflicts for open PRs")
    conf.add_argument("repo", help="<owner>/<repo>")
    conf.add_argument("--token", default=None, help="GitHub token")
    conf.add_argument("--base", default="main", help="Base branch")
    conf.add_argument("--remote", default="origin", help="Remote name")
    conf.add_argument("--repo-path", default=".", help="Local repository path")

    args = parser.parse_args()

    if args.cmd == "monitor":
        if args.use_asyncio:
            async def run() -> None:
                async with aiohttp.ClientSession() as session:
                    prs = await list_open_prs_async(
                        args.repo, args.token, session=session
                    )
                    results = await asyncio.gather(
                        *(
                            check_mergeable_async(
                                args.repo,
                                pr["number"],
                                args.token,
                                session=session,
                            )
                            for pr in prs
                        )
                    )
                    for pr, mergeable in zip(prs, results):
                        print(
                            f"#{pr['number']} {pr['title']} - mergeable: {mergeable}"
                        )
            asyncio.run(run())
        else:
            prs = list_open_prs(args.repo, args.token)
            for pr in prs:
                mergeable = check_mergeable(args.repo, pr["number"], args.token)
                print(f"#{pr['number']} {pr['title']} - mergeable: {mergeable}")
    else:
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
