import unittest
import importlib.machinery
import importlib.util
import types
import sys
import asyncio
import json
from pathlib import Path

pkg = types.ModuleType('asi')
pkg.__path__ = ['src']
sys.modules['asi'] = pkg
sys.modules['PIL'] = types.ModuleType('PIL')
sys.modules['PIL.Image'] = types.ModuleType('PIL.Image')

def load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'asi'
    sys.modules[name] = mod
    loader.exec_module(mod)
    setattr(pkg, name.split('.')[-1], mod)
    return mod

TelemetryLogger = load('asi.telemetry', 'src/telemetry.py').TelemetryLogger
ReasoningHistoryLogger = load('asi.reasoning_history', 'src/reasoning_history.py').ReasoningHistoryLogger
CollaborationPortal = load('asi.collaboration_portal', 'src/collaboration_portal.py').CollaborationPortal
sys.modules['asi.telemetry']._HAS_PROM = False
from aiohttp import ClientSession


class TestCollaborationPortalWS(unittest.TestCase):
    def test_ws(self):
        tel = TelemetryLogger(interval=0.1)
        tel.start()
        rh = ReasoningHistoryLogger()
        portal = CollaborationPortal(['t1'], tel, rh)
        portal.start(port=0)
        portal.start_ws(port=0)
        ws_port = portal.ws_port
        rh.log('m')
        portal.add_task('t2')

        async def run_client():
            assert ws_port is not None
            async with ClientSession() as session:
                async with session.ws_connect(f'http://localhost:{ws_port}/ws') as ws:
                    msg = await ws.receive()
                    data = json.loads(msg.data)
                    return data

        data = asyncio.get_event_loop().run_until_complete(run_client())
        portal.stop_ws()
        portal.stop()
        tel.stop()
        self.assertIn('tasks', data)
        self.assertIn('logs', data)


if __name__ == '__main__':
    unittest.main()
