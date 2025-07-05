import argparse
import time
from asi.graph_of_thought import GraphOfThought
from asi.reasoning_history import ReasoningHistoryLogger
from asi.graph_ui import GraphUI


def main(port: int) -> None:
    graph = GraphOfThought()
    a = graph.add_step("start")
    b = graph.add_step("finish")
    graph.connect(a, b)
    logger = ReasoningHistoryLogger()
    ui = GraphUI(graph, logger)
    ui.start(port=port)
    print(f"Graph UI running at http://localhost:{ui.port}/graph")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    ui.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch editable graph UI")
    parser.add_argument("--port", type=int, default=8070)
    args = parser.parse_args()
    main(args.port)
