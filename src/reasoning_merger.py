from __future__ import annotations

from typing import Mapping, Dict, List, Tuple

from .graph_of_thought import GraphOfThought


def merge_graphs(graphs: Mapping[str, GraphOfThought]) -> Tuple[GraphOfThought, List[Tuple[float, Tuple[str, ...]]]]:
    """Merge multiple ``GraphOfThought`` objects into one.

    Nodes with the same text are deduplicated. Node ids in the merged graph are
    assigned in chronological order based on the ``'timestamp'`` metadata. If
    multiple nodes share a timestamp but have different texts an inconsistency is
    reported.
    """
    all_nodes: List[Tuple[float | None, str, dict, str, int]] = []
    for name, g in graphs.items():
        for nid, node in g.nodes.items():
            ts = node.metadata.get("timestamp")
            all_nodes.append((ts, node.text, dict(node.metadata), name, nid))

    all_nodes.sort(key=lambda x: float("inf") if x[0] is None else float(x[0]))

    merged = GraphOfThought()
    mapping: Dict[Tuple[str, int], int] = {}
    text_map: Dict[str, int] = {}
    ts_map: Dict[float, set[str]] = {}

    for ts, text, meta, gname, nid in all_nodes:
        if ts is not None:
            ts_map.setdefault(float(ts), set()).add(text)
        if text in text_map:
            new_id = text_map[text]
            # keep earliest timestamp
            if ts is not None:
                cur_ts = merged.nodes[new_id].metadata.get("timestamp")
                if cur_ts is None or ts < cur_ts:
                    merged.nodes[new_id].metadata["timestamp"] = ts
        else:
            new_id = merged.add_step(text, metadata={"timestamp": ts})
            text_map[text] = new_id
        mapping[(gname, nid)] = new_id

    for name, g in graphs.items():
        for src, dsts in g.edges.items():
            for dst in dsts:
                merged.connect(mapping[(name, src)], mapping[(name, dst)])

    inconsistencies: List[Tuple[float, Tuple[str, ...]]] = []
    for ts, texts in ts_map.items():
        if len(texts) > 1:
            inconsistencies.append((ts, tuple(sorted(texts))))
    inconsistencies.sort(key=lambda x: x[0])

    return merged, inconsistencies


__all__ = ["merge_graphs"]

