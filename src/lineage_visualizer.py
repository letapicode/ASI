from __future__ import annotations

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from .dataset_lineage_manager import DatasetLineageManager


class LineageVisualizer:
    """Serve an interactive graph of dataset lineage."""

    def __init__(self, manager: DatasetLineageManager) -> None:
        self.manager = manager
        self.httpd: HTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.port: int | None = None
        self._fairness_img: str | None = None

    # --------------------------------------------------------------
    def graph_json(self) -> dict[str, list]:
        nodes: dict[str, dict[str, str]] = {}
        links: list[dict[str, str]] = []
        for step in self.manager.steps:
            for inp in step.inputs:
                nodes[inp] = {"id": inp}
            for out in step.outputs.keys():
                nodes[out] = {"id": out}
                for inp in step.inputs:
                    links.append({"source": inp, "target": out, "note": step.note})
        return {"nodes": list(nodes.values()), "links": links}

    # --------------------------------------------------------------
    def _load_fairness(self) -> None:
        if self._fairness_img is not None:
            return
        fname = f"{Path(self.manager.root).stem}_fairness.png"
        path = Path("docs/datasets") / fname
        if path.exists():
            b64 = base64.b64encode(path.read_bytes()).decode()
            self._fairness_img = f"data:image/png;base64,{b64}"
        else:
            self._fairness_img = ""

    # --------------------------------------------------------------
    def to_html(self) -> str:
        self._load_fairness()
        img = f"<img src='{self._fairness_img}'>" if self._fairness_img else ""
        return (
            "<html><body><h1>Dataset Lineage</h1>" + img +
            "<script src='https://cdn.jsdelivr.net/npm/d3@7'></script>"
            "<svg width='800' height='600'></svg>"
            "<script>"
            "fetch('/graph').then(r=>r.json()).then(data=>{"
            "const svg=d3.select('svg');"
            "const width=+svg.attr('width'),height=+svg.attr('height');"
            "const sim=d3.forceSimulation(data.nodes)"
            ".force('link',d3.forceLink(data.links).id(d=>d.id).distance(40))"
            ".force('charge',d3.forceManyBody().strength(-200))"
            ".force('center',d3.forceCenter(width/2,height/2));"
            "const link=svg.append('g').selectAll('line')"
            ".data(data.links).enter().append('line').attr('stroke','#999');"
            "const node=svg.append('g').selectAll('circle')"
            ".data(data.nodes).enter().append('circle').attr('r',5)"
            ".call(d3.drag()"
            ".on('start',e=>{if(!e.active)sim.alphaTarget(0.3).restart();e.subject.fx=e.subject.x;e.subject.fy=e.subject.y;})"
            ".on('drag',e=>{e.subject.fx=e.x;e.subject.fy=e.y;})"
            ".on('end',e=>{if(!e.active)sim.alphaTarget(0);e.subject.fx=null;e.subject.fy=null;}));"
            "const text=svg.append('g').selectAll('text')"
            ".data(data.nodes).enter().append('text').text(d=>d.id.split('/').pop());"
            "sim.on('tick',()=>{"
            "link.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y)"
            ".attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);"
            "node.attr('cx',d=>d.x).attr('cy',d=>d.y);"
            "text.attr('x',d=>d.x+6).attr('y',d=>d.y+4);"
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
                if self.path == "/graph":
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


__all__ = ["LineageVisualizer"]
