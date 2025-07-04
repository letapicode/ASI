import argparse
import time
from asi.memory_dashboard import MemoryDashboard
from asi.risk_scoreboard import RiskScoreboard
from asi.risk_dashboard import RiskDashboard
from asi.hierarchical_memory import HierarchicalMemory
from asi.memory_service import serve
from asi.telemetry import TelemetryLogger


def main(port: int) -> None:
    mem = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10)
    logger = TelemetryLogger(interval=0.5)
    server = serve(mem, "localhost:50510", telemetry=logger)
    board = RiskScoreboard()
    board.update(0, 0.0, 1.0)
    dash = RiskDashboard(board, [server])
    dash.start(port=port)
    print(f"Dashboard running at http://localhost:{port}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    dash.stop()
    server.stop(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch memory usage dashboard")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()
    main(args.port)
