from pathlib import Path
import sys
import importlib
import pkgutil

# Allow imports without installing the package
_src = Path(__file__).resolve().parent.parent / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

# Import core modules early to avoid circular imports
try:
    hm = importlib.import_module("src.hierarchical_memory")
    globals()["hierarchical_memory"] = hm
    sys.modules[f"{__name__}.hierarchical_memory"] = hm
except Exception:  # pragma: no cover - optional
    pass

# Re-export every module in ``src`` so ``asi.foo`` works in tests
for _mod in pkgutil.iter_modules([str(_src)]):
    try:
        mod = importlib.import_module(f"src.{_mod.name}")
    except Exception:  # pragma: no cover - optional deps may fail
        continue
    globals()[_mod.name] = mod
    sys.modules[f"{__name__}.{_mod.name}"] = mod

from src import *  # noqa: F401,F403
