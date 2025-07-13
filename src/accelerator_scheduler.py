"""Backward compatibility wrapper for :mod:`schedulers`."""

from importlib import import_module

from .scheduler_wrappers import AcceleratorScheduler
psutil = import_module("asi.scheduler_utils").psutil
if psutil is None:
    import types
    psutil = types.SimpleNamespace()

__all__ = ["AcceleratorScheduler"]
