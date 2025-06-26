from pathlib import Path
import sys

# Allow imports without installing the package
_src = Path(__file__).resolve().parent.parent / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from src import *  # noqa: F401,F403
