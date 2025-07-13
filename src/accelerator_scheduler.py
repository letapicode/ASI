"""Backward compatibility wrapper for :mod:`schedulers`."""

from importlib import import_module
from pathlib import Path
import sys

pkg = sys.modules.get("asi")
if pkg is None:
    src_pkg = sys.modules.get("src")
    if src_pkg is not None:
        pkg = sys.modules["asi"] = src_pkg
if pkg is not None and not getattr(pkg, "__path__", None):
    pkg.__path__ = [str(Path(__file__).parent)]

_schedulers = import_module("asi.schedulers")
AcceleratorScheduler = _schedulers.AcceleratorScheduler
psutil = import_module("asi.scheduler_utils").psutil
if psutil is None:
    import types
    psutil = types.SimpleNamespace()

__all__ = ["AcceleratorScheduler"]
