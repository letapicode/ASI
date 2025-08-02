"""Start a dataset lineage dashboard."""
from __future__ import annotations

import argparse
import time

try:  # pragma: no cover - prefer package import
    from asi.dataset_lineage import DatasetLineageManager
    from asi.dataset_lineage_dashboard import DatasetLineageDashboard
except Exception:  # pragma: no cover - fallback for tests
    from src.dataset_lineage import DatasetLineageManager  # type: ignore
    from src.dataset_lineage_dashboard import DatasetLineageDashboard  # type: ignore


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Dataset lineage dashboard")
    parser.add_argument("root", help="Dataset root directory")
    parser.add_argument("--port", type=int, default=8011)
    args = parser.parse_args(argv)

    mgr = DatasetLineageManager(args.root)
    mgr.load()
    dash = DatasetLineageDashboard(mgr)
    dash.start(port=args.port)
    print(f"Serving on http://localhost:{dash.port}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:  # pragma: no cover - manual stop
        pass
    finally:
        dash.stop()


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
