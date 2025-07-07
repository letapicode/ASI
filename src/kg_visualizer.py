from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict

from .knowledge_graph_memory import KnowledgeGraphMemory


class KGVisualizer:
    """Serve an interactive view of a ``KnowledgeGraphMemory``."""

    def __init__(self, kg: KnowledgeGraphMemory) -> None:
        self.kg = kg
        self.httpd: HTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.port: int | None = None

    # --------------------------------------------------------------
    def graph_json(self) -> Dict[str, Any]:
        nodes: Dict[str, Dict[str, str]] = {}
        edges: list[Dict[str, Any]] = []
        for (s, p, o), ts in zip(self.kg.triples, self.kg.timestamps):
            nodes.setdefault(s, {"id": s})
            nodes.setdefault(o, {"id": o})
            edges.append({
                "source": s,
                "target": o,
                "predicate": p,
                "timestamp": ts,
            })
        return {"nodes": list(nodes.values()), "edges": edges}

    # --------------------------------------------------------------
    def to_html(self) -> str:
        return (
            "<html><body><h1>Knowledge Graph</h1>"
            "<script src='https://cdn.jsdelivr.net/npm/d3@7'></script>"
            "<svg width='800' height='600'></svg>"
            "<script>"
            "fetch('/kg/graph').then(r=>r.json()).then(data=>{"
            "const svg=d3.select('svg');"
            "const w=+svg.attr('width'),h=+svg.attr('height');"
            "const sim=d3.forceSimulation(data.nodes)"
            ".force('link',d3.forceLink(data.edges).id(d=>d.id).distance(60))"
            ".force('charge',d3.forceManyBody().strength(-200))"
            ".force('center',d3.forceCenter(w/2,h/2));"
            "const link=svg.append('g').selectAll('line')"
            ".data(data.edges).enter().append('line').attr('stroke','#999');"
            "const node=svg.append('g').selectAll('circle')"
            ".data(data.nodes).enter().append('circle').attr('r',5).attr('fill','#69b');"
            "node.append('title').text(d=>d.id);"
            "const label=svg.append('g').selectAll('text')"
            ".data(data.edges).enter().append('text').text(d=>d.predicate+(d.timestamp?' @'+d.timestamp:''));"
            "sim.on('tick',()=>{"
            "link.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y)"
            ".attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);"
            "node.attr('cx',d=>d.x).attr('cy',d=>d.y);"
            "label.attr('x',d=>(d.source.x+d.target.x)/2)"
            ".attr('y',d=>(d.source.y+d.target.y)/2);"
            "});"
            "});"
            "</script></body></html>"
        )

    # --------------------------------------------------------------
    def start(self, port: int = 8000) -> None:
        if self.httpd is not None:
            return

        visualizer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: D401
                if self.path == "/kg/graph":
                    data = json.dumps(visualizer.graph_json()).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    html = visualizer.to_html().encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(html)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: D401
                return

        self.httpd = HTTPServer(("localhost", port), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    # --------------------------------------------------------------
    def stop(self) -> None:
        if self.httpd is None:
            return
        assert self.thread is not None
        self.httpd.shutdown()
        self.thread.join(timeout=1)
        self.httpd.server_close()
        self.httpd = None
        self.thread = None
        self.port = None


__all__ = ["KGVisualizer"]
