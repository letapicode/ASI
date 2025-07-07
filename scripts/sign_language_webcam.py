"""Stream webcam frames to control a reasoning graph via sign language."""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from asi.graph_of_thought import GraphOfThought
from asi.reasoning_history import ReasoningHistoryLogger
from asi.graph_ui import GraphUI
from asi.voice_graph_controller import SignLanguageGraphController
from asi.data_ingest import CrossLingualTranslator


def main(camera: int, port: int, languages: list[str], mapping: dict[str, str] | None) -> None:
    graph = GraphOfThought()
    a = graph.add_step("start")
    b = graph.add_step("finish")
    graph.connect(a, b)
    logger = ReasoningHistoryLogger()
    ui = GraphUI(graph, logger)
    ui.start(port=port)
    controller = SignLanguageGraphController(
        ui.editor,
        translator=CrossLingualTranslator(languages),
        mapping=mapping,
    )
    cap = cv2.VideoCapture(camera)
    print(f"Graph UI running at http://localhost:{ui.port}/graph")
    try:
        frames: list[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if len(frames) < 16:
                continue
            video = np.stack(frames, axis=0)
            frames.clear()
            try:
                result = controller.apply(video)
                print("result:", result)
            except Exception as exc:
                print("error:", exc)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        ui.stop()


if __name__ == "__main__":  # pragma: no cover - manual demo
    parser = argparse.ArgumentParser(
        description="Control a reasoning graph using sign language"
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--port", type=int, default=8070, help="GraphUI port")
    parser.add_argument(
        "--lang",
        action="append",
        default=["en"],
        help="Supported languages for commands",
    )
    parser.add_argument(
        "--map",
        action="append",
        default=None,
        metavar="GESTURE=CMD",
        help="Map recognized gestures to commands",
    )
    args = parser.parse_args()
    mapping = None
    if args.map:
        mapping = {}
        for pair in args.map:
            if "=" in pair:
                key, cmd = pair.split("=", 1)
                mapping[key] = cmd
    main(args.camera, args.port, args.lang, mapping)
