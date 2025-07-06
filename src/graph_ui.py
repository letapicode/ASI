from __future__ import annotations

import threading
import socket
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from .graph_of_thought import GraphOfThought
from .reasoning_history import ReasoningHistoryLogger


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
<p>Languages: {languages}</p>
<svg width="600" height="400"></svg>
<script>
async function load() {
  const data = await fetch('/graph/data').then(r => r.json());
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
load();
</script>
</body>
</html>
"""


class GraphUI:
    """Serve reasoning graphs and history via FastAPI."""

    def __init__(self, graph: GraphOfThought, logger: ReasoningHistoryLogger) -> None:
        self.graph = graph
        self.logger = logger
        self.app = FastAPI()
        self._setup_routes()
        self.thread: threading.Thread | None = None
        self.server: uvicorn.Server | None = None
        self.port: int | None = None
        # store initial summary
        self.logger.log(self.graph.self_reflect())

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
        async def graph_data() -> Any:
            return JSONResponse(self.graph.to_json())

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
