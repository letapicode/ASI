from __future__ import annotations

from typing import Any, Dict, Sequence

from .graph_of_thought import GraphOfThought
from .data_ingest import CrossLingualTranslator
from .context_summary_memory import ContextSummaryMemory


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
        image_embed: Sequence[float] | None = None,
        audio_embed: Sequence[float] | None = None,
    ) -> int:
        meta = dict(metadata or {})
        meta["lang"] = lang
        if image_embed is not None:
            meta["image_vec"] = list(image_embed)
        if audio_embed is not None:
            meta["audio_vec"] = list(audio_embed)
        return super().add_step(text, meta, node_id)

    def summarize_old_steps(
        self,
        trace: Sequence[int],
        memory: ContextSummaryMemory,
        threshold: int = 10,
    ) -> Sequence[int]:
        """Summarize ``trace`` when it exceeds ``threshold`` nodes."""
        if len(trace) <= threshold:
            return list(trace)
        prefix = trace[:-threshold]
        summary_text = super().summarize_trace(prefix)
        summary = memory.summarizer.summarize(summary_text)
        translator = memory.translator or self.translator
        translations: Dict[str, str] | None = None
        if translator is not None:
            translations = translator.translate_all(summary)
        meta: Dict[str, Any] = {"summary": True}
        if translations is not None:
            meta["translations"] = translations
        nid = self.add_step(summary, metadata=meta)
        return [nid] + list(trace[-threshold:])

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

