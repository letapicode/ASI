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

ARDebugger = importlib.import_module('src.ar_debugger').ARDebugger
SimpleEnv = importlib.import_module('src.self_play_env').SimpleEnv
TrajectoryDataset = importlib.import_module('src.world_model_rl').TrajectoryDataset
RLBridgeConfig = importlib.import_module('src.world_model_rl').RLBridgeConfig
train_world_model = importlib.import_module('src.world_model_rl').train_world_model

from aiohttp import ClientSession
import torch


class TestARDebugger(unittest.TestCase):
    def test_streaming(self):
        env = SimpleEnv(2)
        trans = []
        obs = env.reset()
        for _ in range(2):
            step = env.step(torch.ones(2))
            trans.append((obs.clone(), 0, step.observation.clone(), step.reward))
            obs = step.observation
        cfg = RLBridgeConfig(state_dim=2, action_dim=2, epochs=1, batch_size=1)
        wm = train_world_model(cfg, TrajectoryDataset(trans))
        dbg = ARDebugger()
        dbg.start(port=0)
        port = dbg.port

        async def run_client() -> dict:
            assert port is not None
            async with ClientSession() as session:
                async with session.ws_connect(f'http://localhost:{port}/ws') as ws:
                    pred, _ = wm(obs, torch.tensor(0))
                    dbg.stream_state(pred, obs)
                    msg = await ws.receive()
                    return json.loads(msg.data)

        data = asyncio.get_event_loop().run_until_complete(run_client())
        dbg.stop()
        self.assertIn('predicted', data)
        self.assertIn('actual', data)


if __name__ == '__main__':
    unittest.main()
