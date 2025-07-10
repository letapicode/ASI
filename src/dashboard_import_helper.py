"""Utilities for importing :class:`BaseDashboard` both packaged and from source."""

from importlib import util
from pathlib import Path
import sys
from typing import Type


def load_base_dashboard(current_file: str) -> Type:
    """Return ``BaseDashboard`` class handling local fallback."""
    try:  # try relative import when installed as a package
        from .dashboard_base import BaseDashboard
        return BaseDashboard
    except Exception:
        pass
    spec = util.spec_from_file_location(
        "dashboard_base", Path(current_file).with_name("dashboard_base.py")
    )
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore
    sys.modules.setdefault("dashboard_base", module)
    return module.BaseDashboard  # type: ignore

__all__ = ["load_base_dashboard"]
