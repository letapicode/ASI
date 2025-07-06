from __future__ import annotations

from typing import Iterable, Tuple, Union, Optional, List

from .knowledge_graph_memory import KnowledgeGraphMemory, TimedTriple
from .data_ingest import CrossLingualTranslator


class CrossLingualKGMemory(KnowledgeGraphMemory):
    """Knowledge graph memory that stores translations of triples."""

    def __init__(
        self,
        *args,
        translator: CrossLingualTranslator | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.translator = translator

    # ------------------------------------------------------------
    def add_triples_multilingual(
        self,
        triples: Iterable[Union[Tuple[str, str, str], Tuple[str, str, str, float], TimedTriple]],
    ) -> None:
        """Add triples and their translations."""
        super().add_triples(triples)
        if self.translator is None:
            return
        extra: List[Union[Tuple[str, str, str], Tuple[str, str, str, float]]] = []
        for t in triples:
            ts: Optional[float] = None
            if isinstance(t, TimedTriple):
                s, p, o, ts = t.subject, t.predicate, t.object, t.timestamp
            else:
                s, p, o = t[:3]
                if len(t) == 4:
                    ts = float(t[3])
            for lang in self.translator.languages:
                s_t = self.translator.translate(s, lang)
                p_t = self.translator.translate(p, lang)
                o_t = self.translator.translate(o, lang)
                if ts is not None:
                    extra.append((s_t, p_t, o_t, ts))
                else:
                    extra.append((s_t, p_t, o_t))
        super().add_triples(extra)

    # ------------------------------------------------------------
    def query_translated(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None,
        *,
        lang: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[TimedTriple]:
        """Query triples and optionally translate results and queries."""
        results = self.query_triples(subject, predicate, object, start_time, end_time)
        if self.translator is None:
            return results
        # query translated variants
        queries: List[Tuple[Optional[str], Optional[str], Optional[str]]] = []
        if subject is not None or predicate is not None or object is not None:
            for l in self.translator.languages:
                s = self.translator.translate(subject, l) if subject is not None else None
                p = self.translator.translate(predicate, l) if predicate is not None else None
                o = self.translator.translate(object, l) if object is not None else None
                queries.append((s, p, o))
        for s, p, o in queries:
            results.extend(self.query_triples(s, p, o, start_time, end_time))
        seen = set()
        dedup: List[TimedTriple] = []
        for t in results:
            key = (t.subject, t.predicate, t.object, t.timestamp)
            if key in seen:
                continue
            seen.add(key)
            if lang is not None:
                dedup.append(
                    TimedTriple(
                        self.translator.translate(t.subject, lang),
                        self.translator.translate(t.predicate, lang),
                        self.translator.translate(t.object, lang),
                        t.timestamp,
                    )
                )
            else:
                dedup.append(t)
        return dedup


__all__ = ["CrossLingualKGMemory"]
