from __future__ import annotations

import json
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from typing import Any, Iterable, List, Type

try:
    from .dashboard_import_helper import load_base_dashboard
except Exception:  # pragma: no cover - fallback when not packaged
    import importlib.util
    import sys
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "dashboard_import_helper", Path(__file__).with_name("dashboard_import_helper.py")
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore
    sys.modules.setdefault("dashboard_import_helper", module)
    load_base_dashboard = module.load_base_dashboard  # type: ignore

BaseDashboard = load_base_dashboard(__file__)

from .dataset_lineage import DatasetLineageManager, LineageStep


class DatasetLineageDashboard(BaseDashboard):
    """Serve dataset lineage graphs with basic filtering and search."""

    def __init__(self, manager: DatasetLineageManager) -> None:
        super().__init__()
        self.manager = manager

    # --------------------------------------------------------------
    def _filter_steps(
        self,
        *,
        query: str | None = None,
        note: str | None = None,
        inp: str | None = None,
        out: str | None = None,
    ) -> List[LineageStep]:
        steps = self.manager.steps
        results: List[LineageStep] = []
        for s in steps:
            if note and note not in s.note:
                continue
            if inp and inp not in s.inputs:
                continue
            if out and out not in s.outputs:
                continue
            if query:
                vals = [str(v) for v in s.outputs.values()]
                hay = " ".join([s.note] + s.inputs + list(s.outputs.keys()) + vals)
                if query not in hay:
                    continue
            results.append(s)
        return results

    # --------------------------------------------------------------
    def graph_json(self, steps: Iterable[LineageStep] | None = None) -> dict[str, list]:
        if steps is None:
            steps = self.manager.steps
        nodes: dict[str, dict[str, str]] = {}
        links: list[dict[str, str]] = []
        for step in steps:
            for inp in step.inputs:
                nodes[inp] = {"id": inp}
            for out in step.outputs.keys():
                nodes[out] = {"id": out}
                for inp in step.inputs:
                    links.append({"source": inp, "target": out, "note": step.note})
        return {"nodes": list(nodes.values()), "links": links}

    # --------------------------------------------------------------
    def steps_json(self, steps: Iterable[LineageStep]) -> list[dict[str, Any]]:
        return [asdict(s) for s in steps]

    # --------------------------------------------------------------
    def to_html(self) -> str:
        return (
            "<html><body><h1>Dataset Lineage Dashboard</h1>"
            "<input id='q' placeholder='search'><button onclick='load()'>Search</button>"
            "<script src='https://cdn.jsdelivr.net/npm/d3@7'></script>"
            "<svg width='800' height='600'></svg>"
            "<script>"
            "function draw(data){const svg=d3.select('svg');svg.selectAll('*').remove();"
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
            "});}"
            "function load(){const q=document.getElementById('q').value;"
            "fetch('/graph?q='+encodeURIComponent(q)).then(r=>r.json()).then(draw);}"
            "load();"
            "</script></body></html>"
        )

    # --------------------------------------------------------------
    def get_handler(self) -> Type[BaseHTTPRequestHandler]:
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: D401
                parsed = urlparse(self.path)
                q = parse_qs(parsed.query)
                if parsed.path == "/graph":
                    steps = dashboard._filter_steps(
                        query=q.get("q", [None])[0],
                        note=q.get("note", [None])[0],
                        inp=q.get("input", [None])[0],
                        out=q.get("output", [None])[0],
                    )
                    data = json.dumps(dashboard.graph_json(steps)).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(data)
                elif parsed.path in ("/steps", "/search"):
                    steps = dashboard._filter_steps(
                        query=q.get("q", [None])[0],
                        note=q.get("note", [None])[0],
                        inp=q.get("input", [None])[0],
                        out=q.get("output", [None])[0],
                    )
                    data = json.dumps(dashboard.steps_json(steps)).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    html = dashboard.to_html().encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(html)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: D401
                return

        return Handler

    def start(self, host: str = "localhost", port: int = 8011) -> None:
        super().start(host, port)

    # --------------------------------------------------------------
    def stop(self) -> None:
        super().stop()


__all__ = ["DatasetLineageDashboard"]
