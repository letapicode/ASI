from __future__ import annotations

from importlib import import_module, util
from pathlib import Path
import sys


def _load(name: str):
    """Load ``name`` from :mod:`fairness` with fallback for direct file usage."""
    try:
        mod = import_module(__package__ + '.fairness')
        return getattr(mod, name)
    except Exception:
        path = Path(__file__).with_name('fairness.py')
        spec = util.spec_from_file_location(
            __package__ + '.fairness', path, submodule_search_locations=[str(path.parent)]
        )
        mod = util.module_from_spec(spec)
        sys.modules.setdefault(spec.name, mod)
        if spec.loader:
            spec.loader.exec_module(mod)
        return getattr(mod, name)


FairnessEvaluator = _load('FairnessEvaluator')
CrossLingualFairnessEvaluator = _load('CrossLingualFairnessEvaluator')
FairnessFeedback = _load('FairnessFeedback')
FairnessVisualizer = _load('FairnessVisualizer')
FairnessAdaptationPipeline = _load('FairnessAdaptationPipeline')

__all__ = [
    'FairnessEvaluator',
    'CrossLingualFairnessEvaluator',
    'FairnessFeedback',
    'FairnessAdaptationPipeline',
    'FairnessVisualizer',
]
