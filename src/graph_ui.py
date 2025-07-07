from __future__ import annotations

import threading
import socket
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from .graph_of_thought import GraphOfThought
from .reasoning_history import ReasoningHistoryLogger
from .nl_graph_editor import NLGraphEditor
from .voice_graph_controller import VoiceGraphController
from .cognitive_load_monitor import CognitiveLoadMonitor
from .telemetry import TelemetryLogger


_HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset='utf-8'>
  <title>Reasoning Graph</title>
  <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
</head>
<body>
<h1>Reasoning Graph</h1>

<div>
  <input id='cmd' type='text' placeholder='Edit command'>
  <button onclick='sendCmd()'>Apply</button>
</div>

<div>
  <label for='lang'>Language:</label>
  <select id='lang'></select>
</div>

<svg width="600" height="400"></svg>
<script>
let currentLang = 'en';

async function loadLanguages() {
  const langs = await fetch('/languages').then(r => r.json());
  const sel = document.getElementById('lang');
  sel.innerHTML = '';
  langs.forEach(l => {
    const opt = document.createElement('option');
    opt.value = l;
    opt.text = l;
    sel.appendChild(opt);
  });
  sel.value = currentLang;
  sel.onchange = () => { currentLang = sel.value; load(); };
}

async function load() {
  const data = await fetch('/graph/data?lang=' + currentLang).then(r => r.json());
  const nodes = data.nodes.map(n => ({id: n.id, text: n.text}));
  const links = data.edges.map(e => ({source: e[0], target: e[1]}));
  const svg = d3.select('svg');
  const width = +svg.attr('width');
  const height = +svg.attr('height');
  const simulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).id(d => d.id))
    .force('charge', d3.forceManyBody())
    .force('center', d3.forceCenter(width / 2, height / 2));
  const link = svg.append('g')
    .selectAll('line')
    .data(links)
    .enter().append('line')
    .attr('stroke', '#999');
  const node = svg.append('g')
    .selectAll('circle')
    .data(nodes)
    .enter().append('circle')
    .attr('r', 5)
    .attr('fill', '#69b');
  node.append('title').text(d => d.text);
  simulation.on('tick', () => {
    link.attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);
    node.attr('cx', d => d.x)
        .attr('cy', d => d.y);
  });
}
async function sendCmd() {
  const text = document.getElementById('cmd').value;
  await fetch('/graph/nl_edit', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({command: text, lang: currentLang})
  });
  await load();
}
loadLanguages();
load();
</script>
</body>
</html>
"""


class GraphUI:
    """Serve reasoning graphs and history via FastAPI."""

    def __init__(
        self,
        graph: GraphOfThought,
        logger: ReasoningHistoryLogger,
        *,
        load_monitor: CognitiveLoadMonitor | None = None,
        throttle_threshold: float = 0.7,
        update_interval: float = 1.0,
        telemetry: TelemetryLogger | None = None,
    ) -> None:
        self.graph = graph
        self.logger = logger
        self.editor = NLGraphEditor(graph)
        self.voice = VoiceGraphController(self.editor)
        self.app = FastAPI()
        self.thread: threading.Thread | None = None
        self.server: uvicorn.Server | None = None
        self.port: int | None = None

        self.load_monitor = load_monitor
        self.throttle_threshold = throttle_threshold
        self.update_interval = update_interval
        self.telemetry = telemetry
        self._high_load = False
        self._last_update = 0.0
        self._cached_data: dict | None = None

        if self.load_monitor is not None:
            self.load_monitor.add_callback(self._on_load)
        self._setup_routes()
        # store initial summary
        self.logger.log(self.graph.self_reflect())

    def _on_load(self, load: float) -> None:
        prev = self._high_load
        self._high_load = load >= self.throttle_threshold
        if self._high_load and not prev and self.telemetry is not None:
            self.telemetry.events.append({"event": "ui_throttle", "load": load})

    def _get_graph_json(self, lang: str | None = None) -> dict:
        now = time.time()
        if self._high_load and self._cached_data is not None:
            if now - self._last_update < self.update_interval:
                return self._cached_data
        data = self.graph.to_json()
        if lang and hasattr(self.graph, 'translate_node'):
            for node in data.get("nodes", []):
                try:
                    node["text"] = self.graph.translate_node(node["id"], lang)
                except Exception:
                    pass
        if self._high_load:
            for node in data.get("nodes", []):
                text = node.get("text", "")
                if len(text) > 30:
                    node["text"] = text[:30] + "..."
        self._cached_data = data
        self._last_update = now
        return data

    # --------------------------------------------------------------
    def _setup_routes(self) -> None:
        @self.app.get('/graph', response_class=HTMLResponse)
        async def graph_page() -> Any:
            langs = ''
            if self.logger.translator is not None:
                langs = ', '.join(self.logger.translator.languages)
            return HTMLResponse(_HTML.replace('{languages}', langs))

        async def _record() -> None:
            summary = self.graph.self_reflect()
            self.logger.log(summary)

        @self.app.get('/graph/data')
        async def graph_data(lang: str | None = None) -> Any:
            return JSONResponse(self._get_graph_json(lang))

        @self.app.get('/history')
        async def history() -> Any:
            return JSONResponse(self.logger.get_history())

        @self.app.get('/languages')
        async def languages() -> Any:
            langs = self.logger.translator.languages if self.logger.translator else []
            return JSONResponse(langs)

        @self.app.post('/graph/node')
        async def add_node(req: Request) -> Any:
            data = await req.json()
            lang = data.get('lang')
            if lang and hasattr(self.graph, 'translate_node'):
                node_id = self.graph.add_step(data.get('text', ''), lang=lang, metadata=data.get('metadata'))
            else:
                node_id = self.graph.add_step(data.get('text', ''), data.get('metadata'))
            await _record()
            return JSONResponse({'id': node_id})

        @self.app.post('/graph/edge')
        async def add_edge(req: Request) -> Any:
            data = await req.json()
            self.graph.connect(int(data['src']), int(data['dst']))
            await _record()
            return JSONResponse({'status': 'ok'})

        @self.app.post('/graph/remove_node')
        async def remove_node(req: Request) -> Any:
            data = await req.json()
            node_id = int(data['id'])
            self.graph.nodes.pop(node_id, None)
            self.graph.edges.pop(node_id, None)
            for src, dsts in list(self.graph.edges.items()):
                self.graph.edges[src] = [d for d in dsts if d != node_id]
            await _record()
            return JSONResponse({'status': 'ok'})

        @self.app.post('/graph/remove_edge')
        async def remove_edge(req: Request) -> Any:
            data = await req.json()
            src = int(data['src'])
            dst = int(data['dst'])
            if src in self.graph.edges:
                self.graph.edges[src] = [d for d in self.graph.edges[src] if d != dst]
            await _record()
            return JSONResponse({'status': 'ok'})

        @self.app.post('/graph/nl_edit')
        async def nl_edit(req: Request) -> Any:
            data = await req.json()
            cmd = data.get('command', '')
            lang = data.get('lang')
            before = set(self.graph.nodes)
            try:
                result = self.editor.apply(cmd)
            except Exception as e:
                return JSONResponse({'status': 'error', 'error': str(e)}, status_code=400)
            if lang and hasattr(self.graph, 'translate_node'):
                new_ids = set(self.graph.nodes) - before
                for nid in new_ids:
                    self.graph.nodes[nid].metadata = dict(self.graph.nodes[nid].metadata or {})
                    self.graph.nodes[nid].metadata.setdefault('lang', lang)
            await _record()
            return JSONResponse(result)

        @self.app.post('/graph/voice')
        async def voice(req: Request) -> Any:
            data = await req.json()
            audio = data.get('path') or data.get('audio')
            try:
                result = self.voice.apply(audio)
            except Exception as e:
                return JSONResponse({'status': 'error', 'error': str(e)}, status_code=400)
            await _record()
            return JSONResponse(result)

        @self.app.post('/graph/recompute')
        async def recompute() -> Any:
            summary = self.graph.self_reflect()
            self.logger.log(summary)
            return JSONResponse({'summary': summary})

    # --------------------------------------------------------------
    def start(self, host: str = 'localhost', port: int = 8070) -> None:
        if self.thread is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((host, port))
        _, real_port = sock.getsockname()
        config = uvicorn.Config(self.app, log_level='warning')
        server = uvicorn.Server(config)
        self.server = server
        self.port = real_port

        def run() -> None:
            server.run(sockets=[sock])

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        # allow server to start
        import time
        time.sleep(0.1)

    # --------------------------------------------------------------
    def stop(self) -> None:
        if self.server is None:
            return
        assert self.thread is not None
        self.server.should_exit = True
        self.thread.join(timeout=1.0)
        self.thread = None
        self.server = None
        self.port = None

__all__ = ['GraphUI']
