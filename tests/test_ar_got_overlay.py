import unittest
import asyncio
import importlib
import types
import sys
import json
from pathlib import Path

pkg = types.ModuleType('asi')
sys.modules.setdefault('asi', pkg)
src_pkg = types.ModuleType('src')
src_pkg.__path__ = [str(Path('src'))]
sys.modules.setdefault('src', src_pkg)

ARGOTOverlay = importlib.import_module('src.ar_got_overlay').ARGOTOverlay
GraphOfThought = importlib.import_module('src.graph_of_thought').GraphOfThought

from aiohttp import ClientSession


class TestARGOTOverlay(unittest.TestCase):
    def test_stream(self):
        graph = GraphOfThought()
        a = graph.add_step('start')
        b = graph.add_step('end')
        graph.connect(a, b)
        overlay = ARGOTOverlay(graph)
        overlay.start(port=0)
        port = overlay.port

        async def run_client() -> dict:
            assert port is not None
            async with ClientSession() as session:
                async with session.ws_connect(f'http://localhost:{port}/ws') as ws:
                    overlay.send_graph()
                    msg = await ws.receive()
                    return json.loads(msg.data)

        data = asyncio.get_event_loop().run_until_complete(run_client())
        overlay.stop()
        self.assertIn('nodes', data)
        self.assertIn('edges', data)


if __name__ == '__main__':
    unittest.main()
