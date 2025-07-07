from __future__ import annotations
import argparse
import time

try:  # pragma: no cover - prefer package imports
    from asi.graph_of_thought import GraphOfThought
    from asi.got_3d_visualizer import GOT3DVisualizer, GOT3DViewer
except Exception:  # pragma: no cover - fallback for tests
    from src.graph_of_thought import GraphOfThought  # type: ignore
    from src.got_3d_visualizer import GOT3DVisualizer, GOT3DViewer  # type: ignore


def main(path: str, port: int) -> None:
    graph = GraphOfThought.from_json(path)
    data = graph.to_json()
    vis = GOT3DVisualizer(data.get("nodes", []), [(s, d) for s, d in data.get("edges", [])])
    viewer = GOT3DViewer(vis)
    viewer.start(port=port)
    print(f"3D viewer running at http://localhost:{viewer.port}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    viewer.stop()


if __name__ == "__main__":  # pragma: no cover - entry point
    parser = argparse.ArgumentParser(description="Launch 3D graph viewer")
    parser.add_argument("trace", help="Path to graph JSON trace")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    main(args.trace, args.port)
