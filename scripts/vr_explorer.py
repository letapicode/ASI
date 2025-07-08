from __future__ import annotations
import argparse
import time

try:  # prefer installed package
    from asi.graph_of_thought import GraphOfThought
    from asi.reasoning_history import ReasoningHistoryLogger
    from asi.vr_graph_explorer import VRGraphExplorer
except Exception:  # pragma: no cover - fallback for tests
    from src.graph_of_thought import GraphOfThought  # type: ignore
    from src.reasoning_history import ReasoningHistoryLogger  # type: ignore
    from src.vr_graph_explorer import VRGraphExplorer  # type: ignore


def main(path: str, port: int) -> None:
    graph = GraphOfThought.from_json(path)
    logger = ReasoningHistoryLogger()
    viewer = VRGraphExplorer(graph, logger)
    viewer.start(port=port)
    print(f"VR explorer running at http://localhost:{viewer.port}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    viewer.stop()


if __name__ == "__main__":  # pragma: no cover - CLI entry
    parser = argparse.ArgumentParser(description="Launch VR graph explorer")
    parser.add_argument("trace", help="Path to graph JSON trace")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()
    main(args.trace, args.port)
