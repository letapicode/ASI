from __future__ import annotations

"""Delay dataset downloads based on carbon intensity."""

from dataclasses import dataclass, field
import time
from pathlib import Path
from typing import Iterable, List, Tuple

from .carbon_aware_scheduler import (
    CarbonAwareScheduler,
    get_carbon_intensity,
    get_hourly_forecast,
)
from .data_ingest import download_triples
from .carbon_tracker import CarbonFootprintTracker


@dataclass
class CarbonAwareDatasetIngest:
    """Helper to run ``download_triples`` when the grid is green."""

    scheduler: CarbonAwareScheduler
    check_interval: float | None = None
    max_delay: float = 21600.0
    tracker: CarbonFootprintTracker = field(
        default_factory=lambda: CarbonFootprintTracker(interval=1.0)
    )

    # --------------------------------------------------
    def download_when_green(
        self,
        text_urls: Iterable[str],
        img_urls: Iterable[str],
        audio_urls: Iterable[str],
        out_dir: str,
        **kwargs: object,
    ) -> List[Tuple[Path, Path, Path]]:
        """Wait until intensity ≤ threshold then download."""

        wait = self.check_interval or self.scheduler.check_interval
        self.tracker.start()
        try:
            while True:
                intensity = get_carbon_intensity(self.scheduler.region)
                if intensity <= self.scheduler.threshold:
                    break
                time.sleep(wait)
            return download_triples(
                text_urls,
                img_urls,
                audio_urls,
                out_dir,
                carbon_tracker=self.tracker,
                **kwargs,
            )
        finally:
            self.tracker.stop()

    # --------------------------------------------------
    def download_at_optimal_time(
        self,
        text_urls: Iterable[str],
        img_urls: Iterable[str],
        audio_urls: Iterable[str],
        out_dir: str,
        **kwargs: object,
    ) -> List[Tuple[Path, Path, Path]]:
        """Schedule download at forecasted lowest intensity."""

        self.tracker.start()
        try:
            forecast = get_hourly_forecast(self.scheduler.region)
            delay = 0.0
            if forecast:
                min_idx = int(min(range(len(forecast)), key=lambda i: forecast[i]))
                delay = min_idx * 3600.0
            if delay and delay <= self.max_delay:
                time.sleep(delay)
            return download_triples(
                text_urls,
                img_urls,
                audio_urls,
                out_dir,
                carbon_tracker=self.tracker,
                **kwargs,
            )
        finally:
            self.tracker.stop()


__all__ = ["CarbonAwareDatasetIngest"]

