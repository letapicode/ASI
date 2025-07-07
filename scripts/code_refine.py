"""CLI for ``CodeRefinePipeline``."""

from __future__ import annotations

import argparse
from pathlib import Path

from asi.code_refine import CodeRefinePipeline


def main() -> None:  # pragma: no cover - CLI entry
    parser = argparse.ArgumentParser(description="Refine generated Python code")
    parser.add_argument("path", nargs="+", help="Python files to refine")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print refined code without modifying files",
    )
    args = parser.parse_args()

    for path in args.path:
        file = Path(path)
        source = file.read_text()
        refined = CodeRefinePipeline().refine(source)
        if not args.dry_run:
            file.write_text(refined)
        print(refined)


if __name__ == "__main__":
    main()
