from __future__ import annotations

import base64
import json
import tempfile
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Iterable, Any, Dict, Type

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

from .graph_of_thought import GraphOfThought

try:  # pragma: no cover - optional heavy dep
    import torch
except Exception:  # pragma: no cover - fallback when torch is missing
    torch = None  # type: ignore

from .memory_dashboard import MemoryDashboard
from .transformer_circuits import AttentionVisualizer

BaseDashboard = load_base_dashboard(__file__)

_HTML = """
<!DOCTYPE html>
<html>
<head><meta charset='utf-8'><title>Interpretability Dashboard</title></head>
<body>
<h1>Interpretability Dashboard</h1>
<div id='stats'></div>
<div id='heatmaps'></div>
<script>
async function load() {
  const stats = await fetch('/stats').then(r => r.json());
  document.getElementById('stats').innerText = JSON.stringify(stats);
  const hmaps = await fetch('/heatmaps').then(r => r.json());
  for (const src of hmaps.images) {
    const img = document.createElement('img');
    img.src = src;
    img.style.margin = '2px';
    document.getElementById('heatmaps').appendChild(img);
  }
}
load();
</script>
</body>
</html>
"""


class InterpretabilityDashboard(BaseDashboard):
    """Serve attention heatmaps and memory statistics."""

    def __init__(
        self,
        model: "torch.nn.Module | Any",
        servers: Iterable[Any],
        sample: "torch.Tensor | Any",
        graph: GraphOfThought | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.mem_dash = MemoryDashboard(servers)
        self.graph = graph
        self.tmpdir = Path(tempfile.mkdtemp(prefix="attn_vis_"))
        if torch is not None:
            self.vis = AttentionVisualizer(model, out_dir=str(self.tmpdir))
            self.vis.run(sample)
        else:  # pragma: no cover - torch optional
            self.vis = None

    # --------------------------------------------------------------
    def _heatmaps(self) -> Dict[str, Any]:
        if self.vis is None:
            return {"images": []}
        imgs = []
        for p in self.vis.out_dir.glob("*.png"):
            with open(p, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode()
            imgs.append(f"data:image/png;base64,{b64}")
        return {"images": imgs}

    # --------------------------------------------------------------
    def get_handler(self) -> Type[BaseHTTPRequestHandler]:
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: D401
                if self.path == "/stats":
                    stats = dashboard.mem_dash.aggregate()
                    if dashboard.graph is not None:
                        contribs = {
                            str(nid): node.metadata.get("head_importance")
                            for nid, node in dashboard.graph.nodes.items()
                            if node.metadata and "head_importance" in node.metadata
                        }
                        if contribs:
                            stats["head_contributions"] = contribs
                    data = json.dumps(stats).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(data)
                elif self.path == "/heatmaps":
                    data = json.dumps(dashboard._heatmaps()).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(_HTML.encode())

            def log_message(self, format: str, *args: Any) -> None:  # noqa: D401
                return

        return Handler

    def start(self, host: str = "localhost", port: int = 8060) -> None:
        super().start(host, port)

    # --------------------------------------------------------------
    def stop(self) -> None:
        super().stop()


__all__ = ["InterpretabilityDashboard"]
