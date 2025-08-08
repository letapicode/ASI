import unittest
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path
import json

pkg = types.ModuleType('asi')
sys.modules.setdefault('asi', pkg)
src_pkg = types.ModuleType('src')
src_pkg.__path__ = [str(Path('src'))]
sys.modules.setdefault('src', src_pkg)


def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod

# Stub GraphOfThought to avoid torch dependency
class GraphOfThought:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self._next = 0
    def add_step(self, text):
        nid = self._next
        self._next += 1
        self.nodes[nid] = types.SimpleNamespace(id=nid, text=text, metadata=None)
        self.edges.setdefault(nid, [])
        return nid
    def connect(self, src, dst):
        self.edges.setdefault(src, []).append(dst)
    def to_json(self):
        nodes = [{"id": i, "text": n.text} for i, n in self.nodes.items()]
        edges = [[s, d] for s, ds in self.edges.items() for d in ds]
        return {"nodes": nodes, "edges": edges}

loader = importlib.machinery.SourceFileLoader('graph_visualizer', 'src/graph_visualizer.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
mod = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = mod
loader.exec_module(mod)
GOT3DVisualizer = mod.GOT3DVisualizer
GOT3DViewer = mod.GOT3DViewer

from aiohttp import ClientSession
import asyncio


class TestGOT3DVisualizer(unittest.TestCase):
    def test_html(self):
        nodes = [{"id": "0", "text": "start"}, {"id": "1", "text": "end"}]
        edges = [("0", "1")]
        vis = GOT3DVisualizer(nodes, edges)
        html = vis.to_html()
        self.assertIn("<html", html.lower())

    def test_viewer_stream(self):
        graph = GraphOfThought()
        a = graph.add_step('a')
        b = graph.add_step('b')
        graph.connect(a, b)
        data = graph.to_json()
        vis = GOT3DVisualizer(data['nodes'], [(s, d) for s, d in data['edges']])
        viewer = GOT3DViewer(vis)
        viewer.start(port=0)
        port = viewer.port

        async def run_client() -> str:
            assert port is not None
            async with ClientSession() as sess:
                async with sess.ws_connect(f'http://localhost:{port}/ws') as ws:
                    viewer.send_graph()
                    msg = await ws.receive()
                    return msg.data

        html = asyncio.get_event_loop().run_until_complete(run_client())
        viewer.stop()
        self.assertIn('html', html.lower())


if __name__ == '__main__':
    unittest.main()
