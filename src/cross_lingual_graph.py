from __future__ import annotations

from typing import Any, Dict, Sequence

from .graph_of_thought import GraphOfThought
from .data_ingest import CrossLingualTranslator


class CrossLingualReasoningGraph(GraphOfThought):
    """Graph-of-thought variant that stores language tags."""

    def __init__(self, translator: CrossLingualTranslator | None = None) -> None:
        super().__init__()
        self.translator = translator

    def add_step(
        self,
        text: str,
        lang: str = "en",
        metadata: Dict[str, Any] | None = None,
        node_id: int | None = None,
    ) -> int:
        meta = dict(metadata or {})
        meta["lang"] = lang
        return super().add_step(text, meta, node_id)

    def translate_node(self, node_id: int, target_lang: str) -> str:
        if node_id not in self.nodes:
            raise KeyError("unknown node id")
        node = self.nodes[node_id]
        if target_lang == node.metadata.get("lang"):
            return node.text
        if self.translator is None:
            return node.text
        return self.translator.translate(node.text, target_lang)

    def get_translations(self, node_id: int) -> Dict[str, str]:
        if node_id not in self.nodes:
            raise KeyError("unknown node id")
        node = self.nodes[node_id]
        if self.translator is None:
            return {node.metadata.get("lang", ""): node.text}
        return self.translator.translate_all(node.text)

    def summarize_trace(self, trace: Sequence[int], lang: str | None = None) -> str:
        if lang is None:
            return super().summarize_trace(trace)
        texts = [self.translate_node(n, lang) for n in trace if n in self.nodes]
        return " -> ".join(texts)


__all__ = ["CrossLingualReasoningGraph"]

