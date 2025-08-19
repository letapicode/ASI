from __future__ import annotations

from pathlib import Path
from typing import Callable


def rewrite_text_file(path: str | Path, transform: Callable[[str], str]) -> None:
    """Rewrite a text file in-place using ``transform`` on its contents."""
    p = Path(path)
    p.write_text(transform(p.read_text()))


__all__ = ["rewrite_text_file"]
