#!/usr/bin/env python
"""Demo streaming robot state to an AR client."""

from __future__ import annotations

import time
import torch

try:  # pragma: no cover - prefer package imports
    from asi.ar_debugger import ARDebugger
    from asi.self_play_env import SimpleEnv
    from asi.world_model_rl import (
        RLBridgeConfig,
        train_world_model,
        TrajectoryDataset,
    )
except Exception:  # pragma: no cover - fallback for tests
    from src.ar_debugger import ARDebugger  # type: ignore
    from src.self_play_env import SimpleEnv  # type: ignore
    from src.world_model_rl import (  # type: ignore
        RLBridgeConfig,
        train_world_model,
        TrajectoryDataset,
    )


def main() -> None:
    env = SimpleEnv(2)
    transitions: list[tuple[torch.Tensor, int, torch.Tensor, float]] = []
    obs = env.reset()
    for _ in range(4):
        action = torch.ones(2)
        step = env.step(action)
        transitions.append((obs.clone(), 0, step.observation.clone(), step.reward))
        obs = step.observation

    cfg = RLBridgeConfig(state_dim=2, action_dim=2, epochs=1, batch_size=2)
    wm = train_world_model(cfg, TrajectoryDataset(transitions))

    dbg = ARDebugger()
    dbg.start(port=8765)
    print(f"AR debugger running on ws://localhost:{dbg.port}/ws")

    obs = env.reset()
    for _ in range(10):
        action = torch.ones(2)
        step = env.step(action)
        pred, _ = wm(obs, action)
        dbg.stream_state(pred, step.observation)
        obs = step.observation
        time.sleep(0.1)

    dbg.stop()


if __name__ == "__main__":  # pragma: no cover - demo
    main()
