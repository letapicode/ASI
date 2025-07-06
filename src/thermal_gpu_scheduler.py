"""GPU-aware scheduler that considers device temperature."""

from __future__ import annotations

from .accelerator_scheduler import AcceleratorScheduler
from .telemetry import TelemetryLogger


class ThermalGPUAwareScheduler(AcceleratorScheduler):
    """Dispatch GPU jobs only when the device is cool enough."""

    def __init__(
        self,
        max_temp: float = 80.0,
        *,
        max_mem: float = 0.9,
        check_interval: float = 1.0,
        telemetry: TelemetryLogger | None = None,
    ) -> None:
        super().__init__(max_util=max_mem, check_interval=check_interval)
        self.max_temp = max_temp
        self.telemetry = telemetry or TelemetryLogger(interval=check_interval)

    # --------------------------------------------------
    def _utilization(self, accelerator: str) -> float:  # type: ignore[override]
        util = super()._utilization(accelerator)
        if accelerator == "gpu":
            try:
                if self.telemetry.gpu_temperature() >= self.max_temp:
                    return 1.0
            except Exception:
                pass
        return util


__all__ = ["ThermalGPUAwareScheduler"]
