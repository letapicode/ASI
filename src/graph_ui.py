from __future__ import annotations

import threading
import socket
import time
from typing import Any

try:  # pragma: no cover - optional dependencies
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn
except Exception:  # pragma: no cover - stub if fastapi is missing
    FastAPI = None  # type: ignore[misc]
    Request = object  # type: ignore[misc]
    HTMLResponse = JSONResponse = object  # type: ignore[misc]
    uvicorn = None  # type: ignore[misc]

from .graph_of_thought import GraphOfThought
from .reasoning_history import ReasoningHistoryLogger
from .graph_pruning_manager import GraphPruningManager
from .nl_graph_editor import NLGraphEditor
try:  # pragma: no cover - optional dependency
    from .voice_graph_controller import VoiceGraphController
except Exception:  # pragma: no cover - fallback when missing deps
    class VoiceGraphController:  # type: ignore[dead-code]
        def __init__(self, *_args: Any, **_kw: Any) -> None:
            pass

        def apply(self, _audio: Any) -> Any:
            raise NotImplementedError
try:  # pragma: no cover - optional dependency
    from .cognitive_load_monitor import CognitiveLoadMonitor
except Exception:  # pragma: no cover - fallback
    class CognitiveLoadMonitor:  # type: ignore[dead-code]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def add_callback(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def log_input(self, *_args: Any, **_kwargs: Any) -> None:
            pass

try:  # pragma: no cover - optional dependency
    from .telemetry import TelemetryLogger
except Exception:  # pragma: no cover - fallback
    class TelemetryLogger:  # type: ignore[dead-code]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.events = []

        def start(self) -> None:  # pragma: no cover - stub
            pass

        def stop(self) -> None:  # pragma: no cover - stub
            pass


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
  <input id='searchBox' type='text' placeholder='Search nodes'>
  <button onclick='doSearch()'>Search</button>
</div>
<div id='searchResults'></div>

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

async function doSearch() {
  const q = document.getElementById('searchBox').value;
  const res = await fetch('/graph/search?query=' + encodeURIComponent(q) +
                          '&lang=' + currentLang).then(r => r.json());
  const out = document.getElementById('searchResults');
  out.innerHTML = res.map(r => r.text + ' (id ' + r.id + ')').join('<br>');
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
        pruner: "GraphPruningManager | None" = None,
        prune_threshold: int = 0,
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
        self.pruner = pruner
        self.prune_threshold = prune_threshold
        self._high_load = False
        self._last_update = 0.0
        self._cached_data: dict | None = None
        self._lang_cache: dict[str, dict] = {}

        if self.load_monitor is not None:
            self.load_monitor.add_callback(self._on_load)
        if self.pruner is not None:
            self.pruner.attach(self.graph)
        self._setup_routes()
        # store initial summary
        self.logger.log(self.graph.self_reflect())

    def _invalidate_cache(self) -> None:
        self._cached_data = None
        self._lang_cache.clear()

    def _on_load(self, load: float) -> None:
        prev = self._high_load
        self._high_load = load >= self.throttle_threshold
        if self._high_load and not prev and self.telemetry is not None:
            self.telemetry.events.append({"event": "ui_throttle", "load": load})

    def _get_graph_json(self, lang: str | None = None) -> dict:
        now = time.time()
        base = self._cached_data
        if (
            base is None
            or not self._high_load
            or now - self._last_update >= self.update_interval
        ):
            base = self.graph.to_json()
            if self._high_load:
                for node in base.get("nodes", []):
                    text = node.get("text", "")
                    if len(text) > 30:
                        node["text"] = text[:30] + "..."
            self._cached_data = base
            self._lang_cache.clear()
            self._last_update = now

        if not lang or lang == "en":
            return base

        cached = self._lang_cache.get(lang)
        if cached is not None:
            return cached

        data = {"nodes": [dict(n) for n in base.get("nodes", [])], "edges": base.get("edges", [])}
        if hasattr(self.graph, "translate_node"):
            for node in data["nodes"]:
                try:
                    node["text"] = self.graph.translate_node(node["id"], lang)
                except Exception:
                    pass
        self._lang_cache[lang] = data
        return data

    def _search_nodes(self, query: str, lang: str = "en") -> list[dict]:
        translator = getattr(self.graph, "translator", None)
        queries = [query.lower()]
        if translator is not None:
            try:
                for t in translator.translate_all(query).values():
                    queries.append(t.lower())
            except Exception:
                pass
        results = []
        for nid, node in self.graph.nodes.items():
            text = node.text.lower()
            if any(q in text for q in queries):
                if hasattr(self.graph, "translate_node"):
                    try:
                        disp = self.graph.translate_node(nid, lang)
                    except Exception:
                        disp = node.text
                else:
                    disp = node.text
                results.append({"id": nid, "text": disp})
        return results

    # --------------------------------------------------------------
    def _setup_routes(self) -> None:
        @self.app.get('/graph', response_class=HTMLResponse)
        async def graph_page() -> Any:
            return HTMLResponse(_HTML)

        async def _record() -> None:
            summary = self.graph.self_reflect()
            self.logger.log(summary)
            if self.pruner is not None and len(self.graph.nodes) > self.prune_threshold:
                self.pruner.prune_low_degree()
                self.pruner.prune_old_nodes()

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

        @self.app.get('/graph/search')
        async def graph_search(query: str, lang: str = 'en') -> Any:
            return JSONResponse(self._search_nodes(query, lang))

        @self.app.post('/graph/node')
        async def add_node(req: Request) -> Any:
            data = await req.json()
            lang = data.get('lang')
            if lang and hasattr(self.graph, 'translate_node'):
                node_id = self.graph.add_step(
                    data.get('text', ''), lang=lang, metadata=data.get('metadata')
                )
            else:
                node_id = self.graph.add_step(
                    data.get('text', ''), data.get('metadata')
                )
            self._invalidate_cache()
            await _record()
            return JSONResponse({'id': node_id})

        @self.app.post('/graph/edge')
        async def add_edge(req: Request) -> Any:
            data = await req.json()
            self.graph.connect(int(data['src']), int(data['dst']))
            self._invalidate_cache()
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
            self._invalidate_cache()
            await _record()
            return JSONResponse({'status': 'ok'})

        @self.app.post('/graph/remove_edge')
        async def remove_edge(req: Request) -> Any:
            data = await req.json()
            src = int(data['src'])
            dst = int(data['dst'])
            if src in self.graph.edges:
                self.graph.edges[src] = [d for d in self.graph.edges[src] if d != dst]
            self._invalidate_cache()
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
                    self.graph.nodes[nid].metadata = dict(
                        self.graph.nodes[nid].metadata or {}
                    )
                    self.graph.nodes[nid].metadata.setdefault('lang', lang)
            self._invalidate_cache()
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
            self._invalidate_cache()
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
