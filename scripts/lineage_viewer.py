"""Start a web server to visualize dataset lineage."""
from __future__ import annotations

import argparse
import time

try:  # pragma: no cover - prefer package import
    from asi.dataset_lineage import DatasetLineageManager
    from asi.graph_visualizers import LineageVisualizer
except Exception:  # pragma: no cover - fallback for tests
    from src.dataset_lineage import DatasetLineageManager  # type: ignore
    from src.graph_visualizers import LineageVisualizer  # type: ignore


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Dataset lineage viewer")
    parser.add_argument("root", help="Dataset root directory")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args(argv)

    mgr = DatasetLineageManager(args.root)
    mgr.load()
    viz = LineageVisualizer(mgr)
    viz.start(port=args.port)
    print(f"Serving on http://localhost:{viz.port}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:  # pragma: no cover - manual stop
        pass
    finally:
        viz.stop()


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
