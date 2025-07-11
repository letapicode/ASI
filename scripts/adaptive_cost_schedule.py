"""Experiment with AdaptiveCostScheduler for multiple clusters."""

from __future__ import annotations

import argparse
from asi.hpc_base_scheduler import make_scheduler
from asi.adaptive_cost_scheduler import AdaptiveCostScheduler


def main() -> None:  # pragma: no cover - CLI entry
    parser = argparse.ArgumentParser(description="Adaptive cost scheduling demo")
    parser.add_argument("command", nargs='+', help="Command to submit")
    parser.add_argument("--qtable", type=str, help="Path to Q-table")
    parser.add_argument("--bins", type=int, default=10, help="Number of score buckets")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Exploration rate")
    parser.add_argument("--alpha", type=float, default=0.5, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    parser.add_argument("--check-interval", type=float, default=60.0,
                        help="Delay between checks when waiting")
    args = parser.parse_args()

    clusters = {
        "east": make_scheduler('arima'),
        "west": make_scheduler('arima', backend="k8s"),
    }
    sched = AdaptiveCostScheduler(
        clusters,
        bins=args.bins,
        epsilon=args.epsilon,
        alpha=args.alpha,
        gamma=args.gamma,
        check_interval=args.check_interval,
        qtable_path=args.qtable,
    )
    cluster, job_id = sched.submit_best(args.command)
    print(f"{cluster} -> {job_id}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
