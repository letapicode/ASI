from __future__ import annotations

from importlib import import_module, util
from pathlib import Path
import sys

try:
    FairnessEvaluator = import_module(__package__ + '.fairness').FairnessEvaluator
except Exception:  # pragma: no cover - fallback for direct file loading
    path = Path(__file__).with_name('fairness.py')
    spec = util.spec_from_file_location(
        __package__ + '.fairness', path, submodule_search_locations=[str(path.parent)]
    )
    mod = util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, mod)
    if spec.loader:
        spec.loader.exec_module(mod)
    FairnessEvaluator = mod.FairnessEvaluator

__all__ = ["FairnessEvaluator"]
