from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys

pkg = sys.modules.get('asi')
if pkg is None:
    src_pkg = sys.modules.get('src')
    if src_pkg is not None:
        pkg = sys.modules['asi'] = src_pkg
if pkg is not None and not getattr(pkg, '__path__', None):
    pkg.__path__ = [str(Path(__file__).parent)]

_sched = import_module('asi.schedulers')

AcceleratorScheduler = _sched.AcceleratorScheduler
AdaptiveScheduler = _sched.AdaptiveScheduler
EnergyAwareScheduler = _sched.EnergyAwareScheduler
BatteryAwareScheduler = _sched.BatteryAwareScheduler

__all__ = [
    'AcceleratorScheduler',
    'AdaptiveScheduler',
    'EnergyAwareScheduler',
    'BatteryAwareScheduler',
]
