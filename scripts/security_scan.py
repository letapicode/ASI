#!/usr/bin/env python
"""Run dependency and source-code security scans."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def run_tool(tool: str, args: list[str]) -> int:
    """Return exit code after running ``tool`` with ``args``.

    Prints a warning if the tool is not installed.
    """
    if shutil.which(tool) is None:
        print(f"{tool} not found; install it with `pip install {tool}`")
        return 1
    cmd = [tool] + args
    print("$", " ".join(cmd))
    return subprocess.call(cmd)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    requirements = repo_root / "requirements.txt"
    src_dir = repo_root / "src"

    status = 0
    status |= run_tool("pip-audit", ["-r", str(requirements)])
    status |= run_tool("bandit", ["-r", str(src_dir)])
    sys.exit(status)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
