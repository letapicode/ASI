#!/usr/bin/env python
"""Demo streaming robot state to an AR client."""

import argparse

from __future__ import annotations

import time
import torch

try:  # pragma: no cover - prefer package imports
    from asi.ar_debugger import ARDebugger
    from asi.ar_got_overlay import ARGOTOverlay
    from asi.graph_of_thought import GraphOfThought
    from asi.self_play_env import SimpleEnv
    from asi.world_model_rl import (
        RLBridgeConfig,
        train_world_model,
        TrajectoryDataset,
    )
except Exception:  # pragma: no cover - fallback for tests
    from src.ar_debugger import ARDebugger  # type: ignore
    from src.ar_got_overlay import ARGOTOverlay  # type: ignore
    from src.graph_of_thought import GraphOfThought  # type: ignore
    from src.self_play_env import SimpleEnv  # type: ignore
    from src.world_model_rl import (  # type: ignore
        RLBridgeConfig,
        train_world_model,
        TrajectoryDataset,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="AR robot demo")
    parser.add_argument("--show-graph", action="store_true", help="Stream reasoning graph")
    args = parser.parse_args(argv)

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

    overlay: ARGOTOverlay | None = None
    graph: GraphOfThought | None = None
    last_node: int | None = None
    if args.show_graph:
        graph = GraphOfThought()
        last_node = graph.add_step("start")
        overlay = ARGOTOverlay(graph)
        overlay.start(port=8766)
        print(f"Graph overlay running on ws://localhost:{overlay.port}/ws")

    obs = env.reset()
    for i in range(10):
        action = torch.ones(2)
        step = env.step(action)
        pred, _ = wm(obs, action)
        dbg.stream_state(pred, step.observation)
        if args.show_graph and overlay is not None and graph is not None and last_node is not None:
            node = graph.add_step(f"step {i}")
            graph.connect(last_node, node)
            last_node = node
            overlay.send_graph()
        obs = step.observation
        time.sleep(0.1)

    dbg.stop()
    if overlay is not None:
        overlay.stop()


if __name__ == "__main__":  # pragma: no cover - demo
    main()
