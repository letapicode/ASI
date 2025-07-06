import argparse
import time

from asi.graph_of_thought import GraphOfThought
from asi.reasoning_history import ReasoningHistoryLogger
from asi.telemetry import TelemetryLogger
from asi.introspection_dashboard import IntrospectionDashboard


def main(port: int) -> None:
    graph = GraphOfThought()
    a = graph.add_step("start", metadata={"timestamp": 0.0})
    b = graph.add_step("finish", metadata={"timestamp": 1.0})
    graph.connect(a, b)

    history = ReasoningHistoryLogger()
    history.log(graph.self_reflect())

    telemetry = TelemetryLogger(interval=0.5)
    telemetry.start()

    dash = IntrospectionDashboard(graph, history, telemetry)
    dash.start(port=port)
    print(f"Introspection dashboard running at http://localhost:{dash.port}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    dash.stop()
    telemetry.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch introspection dashboard")
    parser.add_argument("--port", type=int, default=8060)
    args = parser.parse_args()
    main(args.port)

