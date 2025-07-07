"""Submit a job to the best HPC cluster based on forecasted cost and carbon."""

from __future__ import annotations

import argparse
from asi.hpc_forecast_scheduler import HPCForecastScheduler
from asi.hpc_multi_scheduler import MultiClusterScheduler


def main() -> None:  # pragma: no cover - CLI entry
    parser = argparse.ArgumentParser(description="Multi-cluster scheduling demo")
    parser.add_argument("command", nargs='+', help="Command to submit")
    args = parser.parse_args()

    clusters = {
        "east": HPCForecastScheduler(),
        "west": HPCForecastScheduler(backend="k8s"),
    }
    sched = MultiClusterScheduler(clusters)
    cluster, job_id = sched.submit_best(args.command)
    print(f"{cluster} -> {job_id}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
