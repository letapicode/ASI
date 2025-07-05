from __future__ import annotations

import json
import math
from typing import Any, Dict, Iterable, List, Tuple

import plotly.graph_objects as go


class GOTVisualizer:
    """Render reasoning graphs with Plotly."""

    def __init__(self, nodes: Iterable[Dict[str, Any]], edges: Iterable[Tuple[str, str]]) -> None:
        self.nodes = list(nodes)
        self.edges = list(edges)

    @classmethod
    def from_json(cls, path: str) -> "GOTVisualizer":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        nodes = data.get("nodes", [])
        raw_edges = data.get("edges", [])
        if raw_edges and isinstance(raw_edges[0], dict):
            edges = [(e["source"], e["target"]) for e in raw_edges]
        else:
            edges = [(src, dst) for src, dst in raw_edges]
        return cls(nodes, edges)

    # --------------------------------------------------------------
    def _layout(self) -> Dict[str, Tuple[float, float]]:
        n = max(len(self.nodes), 1)
        pos: Dict[str, Tuple[float, float]] = {}
        for i, node in enumerate(self.nodes):
            angle = 2 * math.pi * i / n
            pos[str(node["id"])] = (math.cos(angle), math.sin(angle))
        return pos

    # --------------------------------------------------------------
    def to_figure(self) -> go.Figure:
        pos = self._layout()
        edge_x: List[float] = []
        edge_y: List[float] = []
        for src, dst in self.edges:
            x0, y0 = pos.get(str(src), (0, 0))
            x1, y1 = pos.get(str(dst), (0, 0))
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(color="#888", width=1),
            hoverinfo="none",
        )
        node_x = []
        node_y = []
        texts = []
        for node in self.nodes:
            nid = str(node["id"])
            x, y = pos[nid]
            node_x.append(x)
            node_y.append(y)
            texts.append(node.get("text", nid))
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=texts,
            textposition="bottom center",
            marker=dict(size=10, color="#1f77b4"),
        )
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig

    # --------------------------------------------------------------
    def to_html(self, title: str = "Reasoning Graph") -> str:
        fig = self.to_figure()
        fig.update_layout(title=title)
        return fig.to_html(include_plotlyjs="cdn")


__all__ = ["GOTVisualizer"]
