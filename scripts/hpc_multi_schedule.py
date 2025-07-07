"""Submit a job to the best HPC cluster based on forecasted cost and carbon."""

from __future__ import annotations

import argparse
from asi.hpc_forecast_scheduler import HPCForecastScheduler
from asi.hpc_multi_scheduler import MultiClusterScheduler
from asi.rl_cost_scheduler import RLCostScheduler


def main() -> None:  # pragma: no cover - CLI entry
    parser = argparse.ArgumentParser(description="Multi-cluster scheduling demo")
    parser.add_argument("command", nargs='+', help="Command to submit")
    parser.add_argument("--rl-cost", action="store_true", help="Use RL cost scheduler")
    args = parser.parse_args()

    clusters = {
        "east": HPCForecastScheduler(),
        "west": HPCForecastScheduler(backend="k8s"),
    }
    if args.rl_cost:
        sched = RLCostScheduler(clusters)
    else:
        sched = MultiClusterScheduler(clusters)
    cluster, job_id = sched.submit_best(args.command)
    print(f"{cluster} -> {job_id}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
