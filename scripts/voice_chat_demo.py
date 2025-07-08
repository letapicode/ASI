"""Launch a simple cross-lingual voice chat agent."""

from __future__ import annotations

import argparse
import time

from asi.data_ingest import CrossLingualTranslator, CrossLingualSpeechTranslator
from asi.graph_of_thought import GraphOfThought
from asi.reasoning_history import ReasoningHistoryLogger
from asi.cross_lingual_voice_chat import CrossLingualVoiceChat
from asi.graph_ui import GraphUI


def main(port: int, languages: list[str]) -> None:
    translator = CrossLingualTranslator(languages)
    speech = CrossLingualSpeechTranslator(translator)
    graph = GraphOfThought()
    logger = ReasoningHistoryLogger(translator=translator)
    chat = CrossLingualVoiceChat(speech, graph)
    ui = GraphUI(graph, logger, voice_chat=chat)
    ui.start(port=port)
    print(f"Voice chat running at http://localhost:{ui.port}/graph")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        ui.stop()


if __name__ == "__main__":  # pragma: no cover - demo helper
    parser = argparse.ArgumentParser(description="Cross-lingual voice chat demo")
    parser.add_argument("--port", type=int, default=8070, help="GraphUI port")
    parser.add_argument("--lang", action="append", default=["en"], help="Supported languages")
    args = parser.parse_args()
    main(args.port, args.lang)
