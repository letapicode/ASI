#!/usr/bin/env python
"""Demo of the ResourceBroker."""

from __future__ import annotations

import random
from asi.resource_broker import ResourceBroker


def main() -> None:
    broker = ResourceBroker()
    broker.register_cluster("cloud", 4)
    broker.register_cluster("on_prem", 2)
    for i in range(3):
        metrics = {"cpu": random.randint(20, 90), "gpu": random.randint(20, 90)}
        decision = broker.scale_decision(metrics)
        cluster = broker.allocate(f"job{i}")
        print(f"job{i} -> {cluster}, decision={decision}, metrics={metrics}")


if __name__ == "__main__":  # pragma: no cover - demo
    main()
