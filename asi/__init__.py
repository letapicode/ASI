from pathlib import Path
import sys
import importlib
import pkgutil

# Allow imports without installing the package
_src = Path(__file__).resolve().parent.parent / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

# Re-export every module in ``src`` so ``asi.foo`` works in tests
for _mod in pkgutil.iter_modules([str(_src)]):
    mod = importlib.import_module(f"src.{_mod.name}")
    globals()[_mod.name] = mod
    sys.modules[f"{__name__}.{_mod.name}"] = mod

from src import *  # noqa: F401,F403
