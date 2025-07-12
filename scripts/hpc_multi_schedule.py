"""Submit a job to the best HPC cluster based on forecasted cost and carbon."""

from __future__ import annotations

import argparse
from asi.hpc_schedulers import make_scheduler, MultiClusterScheduler
from asi.rl_schedulers import RLCarbonScheduler, RLCostScheduler
from asi.carbon_aware_scheduler import CarbonAwareScheduler
from asi.meta_scheduler import MetaScheduler


def main() -> None:  # pragma: no cover - CLI entry
    parser = argparse.ArgumentParser(description="Multi-cluster scheduling demo")
    parser.add_argument("command", nargs='+', help="Command to submit")
    parser.add_argument("--rl-cost", action="store_true", help="Use RL cost scheduler")
    parser.add_argument("--meta", action="store_true", help="Use meta scheduler")
    args = parser.parse_args()

    clusters = {
        "east": make_scheduler('arima'),
        "west": make_scheduler('arima', backend="k8s"),
    }
    if args.meta:
        scheds = {
            "carbon": CarbonAwareScheduler(0.5),
            "rl": RLCarbonScheduler([], telemetry=None),
            "forecast": make_scheduler('arima'),
        }
        sched = MetaScheduler(scheds)
    elif args.rl_cost:
        sched = RLCostScheduler(clusters)
    else:
        sched = MultiClusterScheduler(clusters)
    cluster, job_id = sched.submit_best(args.command)
    print(f"{cluster} -> {job_id}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
