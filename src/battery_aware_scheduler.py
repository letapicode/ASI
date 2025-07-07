from __future__ import annotations

import psutil

from .adaptive_scheduler import AdaptiveScheduler


class BatteryAwareScheduler(AdaptiveScheduler):
    """Pause jobs when system battery level is low."""

    def __init__(
        self,
        budget,
        run_id: str,
        max_mem: float = 0.9,
        check_interval: float = 1.0,
        window: int = 3,
        min_improvement: float = 0.01,
        battery_threshold: float = 0.2,
    ) -> None:
        super().__init__(
            budget,
            run_id,
            max_mem=max_mem,
            check_interval=check_interval,
            window=window,
            min_improvement=min_improvement,
        )
        self.battery_threshold = battery_threshold

    # --------------------------------------------------------------
    def _battery_level(self) -> float:
        try:
            info = psutil.sensors_battery()
            if info is not None and info.percent is not None:
                level = float(info.percent) / 100.0
                self.telemetry.metrics["battery"] = info.percent
                return level
        except Exception:
            pass
        self.telemetry.metrics["battery"] = 100.0
        return 1.0

    # --------------------------------------------------------------
    def _should_pause(self) -> bool:  # type: ignore[override]
        if self._battery_level() < self.battery_threshold:
            return True
        return super()._should_pause()


__all__ = ["BatteryAwareScheduler"]
