import unittest
import asyncio
import json
from datetime import datetime
import importlib
import types
import sys
from pathlib import Path

asi_pkg = types.ModuleType('asi')
sys.modules.setdefault('asi', asi_pkg)
src_pkg = types.ModuleType('src')
src_pkg.__path__ = [str(Path('src'))]
sys.modules.setdefault('src', src_pkg)

CollaborationPortal = importlib.import_module('src.collaboration_portal').CollaborationPortal
ReasoningHistoryLogger = importlib.import_module('src.reasoning_history').ReasoningHistoryLogger
from aiohttp import ClientSession


class TestCollaborationPortalWS(unittest.TestCase):
    def test_ws_updates(self):
        rh = ReasoningHistoryLogger()
        portal = CollaborationPortal([], None, rh)
        portal.start(port=0, ws_port=0)
        ws_port = portal.ws_port
        self.assertIsNotNone(ws_port)

        async def run_client():
            assert ws_port is not None
            async with ClientSession() as session:
                async with session.ws_connect(f'http://localhost:{ws_port}/ws') as ws:
                    await ws.receive()  # initial state
                    await ws.send_str(json.dumps({
                        'type': 'add_task',
                        'task': 'demo',
                        'ts': datetime.utcnow().isoformat()
                    }))
                    msg = await ws.receive()
                    data = json.loads(msg.data)
                    await ws.send_str(json.dumps({
                        'type': 'log',
                        'message': 'done',
                        'ts': datetime.utcnow().isoformat()
                    }))
                    msg2 = await ws.receive()
                    data2 = json.loads(msg2.data)
                    return data, data2

        data1, data2 = asyncio.get_event_loop().run_until_complete(run_client())
        portal.stop()

        self.assertIn('demo', data1['tasks'])
        self.assertEqual(data2['logs'][-1][1], 'done')


if __name__ == '__main__':  # pragma: no cover - test helper
    unittest.main()
