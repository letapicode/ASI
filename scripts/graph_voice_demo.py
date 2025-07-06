"""Demo using voice commands to edit a reasoning graph."""

from __future__ import annotations

import argparse
import http.client
import json
import time

from asi.graph_of_thought import GraphOfThought
from asi.reasoning_history import ReasoningHistoryLogger
from asi.graph_ui import GraphUI


def main(audio_path: str, port: int) -> None:
    graph = GraphOfThought()
    a = graph.add_step("start")
    b = graph.add_step("finish")
    graph.connect(a, b)
    logger = ReasoningHistoryLogger()
    ui = GraphUI(graph, logger)
    ui.start(port=port)
    conn = http.client.HTTPConnection("localhost", ui.port)
    body = json.dumps({"path": audio_path})
    conn.request("POST", "/graph/voice", body, {"Content-Type": "application/json"})
    resp = conn.getresponse()
    print("response:", resp.read().decode())
    print(f"Graph UI running at http://localhost:{ui.port}/graph")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    ui.stop()


if __name__ == "__main__":  # pragma: no cover - demo
    parser = argparse.ArgumentParser(description="Graph voice demo")
    parser.add_argument("audio", help="path to WAV file containing a command")
    parser.add_argument("--port", type=int, default=8070)
    args = parser.parse_args()
    main(args.audio, args.port)
