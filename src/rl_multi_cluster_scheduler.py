from __future__ import annotations

"""Reinforcement learning based multi-cluster HPC scheduler."""

import random
import time
from dataclasses import dataclass, field
from typing import Dict, Tuple, Union, List

from .hpc_forecast_scheduler import HPCForecastScheduler
from .hpc_multi_scheduler import MultiClusterScheduler
from .hpc_scheduler import submit_job


@dataclass
class RLMultiClusterScheduler(MultiClusterScheduler):
    """Choose HPC clusters with a simple Q-learning policy."""

    alpha: float = 0.5
    epsilon: float = 0.1
    q_table: Dict[Tuple[str, int], float] = field(default_factory=dict)
    last_queue: Dict[str, float] = field(default_factory=dict)

    # --------------------------------------------------
    def update_policy(self, log_entry: Dict[str, float | str]) -> None:
        """Update Q-values from a completed job log."""
        cluster = str(log_entry.get("cluster"))
        hour = int(log_entry.get("hour", 0)) % 24
        queue = float(log_entry.get("queue_time", 0.0))
        duration = float(log_entry.get("duration", 0.0))
        carbon = float(log_entry.get("carbon", 0.0))
        reward = -(queue + duration) - carbon * duration
        key = (cluster, hour)
        old = self.q_table.get(key, 0.0)
        self.q_table[key] = old + self.alpha * (reward - old)
        self.last_queue[cluster] = queue

    # --------------------------------------------------
    def submit_best_rl(
        self,
        command: Union[str, List[str]],
        max_delay: float = 21600.0,
        *,
        expected_duration: float = 1.0,
    ) -> Tuple[str, str]:
        """Return chosen cluster name and job id using the RL policy."""

        hour = int(time.time() // 3600) % 24
        clusters = list(self.clusters.keys())
        if random.random() < self.epsilon or not self.q_table:
            choice = random.choice(clusters)
            backend = self.clusters[choice].backend
            tel = self.telemetry.get(choice) if self.telemetry else None
            job_id = submit_job(command, backend=backend, telemetry=tel)
            if tel is not None and self.telemetry:
                baseline = sum(
                    t.get_live_carbon_intensity() for t in self.telemetry.values()
                ) / len(self.telemetry)
                chosen = tel.get_live_carbon_intensity()
                saving = (baseline - chosen) * expected_duration
                tel.metrics["carbon_saved"] = tel.metrics.get("carbon_saved", 0.0) + saving
                self.schedule_log.append((choice, saving))
                if self.dashboard is not None:
                    self.dashboard.record_schedule(choice, saving)
            return choice, job_id

        best_cluster = clusters[0]
        best_val = self.q_table.get((best_cluster, hour), float("-inf"))
        for name in clusters[1:]:
            val = self.q_table.get((name, hour), float("-inf"))
            if val > best_val:
                best_val = val
                best_cluster = name
        backend = self.clusters[best_cluster].backend
        tel = self.telemetry.get(best_cluster) if self.telemetry else None
        job_id = submit_job(command, backend=backend, telemetry=tel)
        if tel is not None and self.telemetry:
            baseline = sum(
                t.get_live_carbon_intensity() for t in self.telemetry.values()
            ) / len(self.telemetry)
            chosen = tel.get_live_carbon_intensity()
            saving = (baseline - chosen) * expected_duration
            tel.metrics["carbon_saved"] = tel.metrics.get("carbon_saved", 0.0) + saving
            self.schedule_log.append((best_cluster, saving))
            if self.dashboard is not None:
                self.dashboard.record_schedule(best_cluster, saving)
        return best_cluster, job_id


__all__ = ["RLMultiClusterScheduler"]
