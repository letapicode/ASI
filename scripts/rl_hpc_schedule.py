"""Train and simulate the RL multi-cluster scheduler."""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List

from asi.hpc_base_scheduler import make_scheduler
from asi.rl_schedulers import RLMultiClusterScheduler


def load_history(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = {
                "cluster": path.stem,
                "hour": int(r.get("hour", 0)),
                "queue_time": float(r.get("queue_time", 0.0)),
                "spot_price": float(r.get("spot_price", 0.0)),
                "carbon": float(r.get("carbon", 0.0)),
                "duration": float(r.get("duration", 1.0)),
            }
            rows.append(row)
    return rows


def main() -> None:  # pragma: no cover - CLI entry
    parser = argparse.ArgumentParser(description="RL HPC scheduling demo")
    parser.add_argument(
        "clusters", nargs="+", help="Pairs of name=csv containing historical data"
    )
    args = parser.parse_args()

    schedulers: Dict[str, object] = {}
    history: List[Dict[str, float]] = []
    for pair in args.clusters:
        name, path = pair.split("=", 1)
        csv_path = Path(path)
        schedulers[name] = make_scheduler('arima')
        for entry in load_history(csv_path):
            entry["cluster"] = name
            history.append(entry)

    rl_sched = RLMultiClusterScheduler(schedulers)
    for entry in history:
        rl_sched.update_policy(entry)

    total_cost = 0.0
    total_carbon = 0.0
    for h in range(24):
        fake_time = h * 3600.0
        with patch_time(fake_time):
            cluster, _ = rl_sched.submit_best_rl(["job.sh"])
        entry = next(e for e in history if e["cluster"] == cluster and e["hour"] == h)
        total_cost += entry["spot_price"] * entry["duration"]
        total_carbon += entry["carbon"] * entry["duration"]
    print(f"Total cost: {total_cost:.2f} \u2022 Emissions: {total_carbon:.2f}")


class patch_time:
    def __init__(self, value: float) -> None:
        self.value = value
        self.old = time.time

    def __enter__(self):
        time.time = lambda: self.value

    def __exit__(self, exc_type, exc, tb):
        time.time = self.old


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
