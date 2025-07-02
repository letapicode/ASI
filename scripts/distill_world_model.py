#!/usr/bin/env python
"""Train a student world model from a larger teacher."""

from pathlib import Path
import argparse
import torch

from asi.multimodal_world_model import MultiModalWorldModel, MultiModalWorldModelConfig, TrajectoryDataset
from asi.world_model_distiller import DistillConfig, distill_world_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill world model")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    cfg_t = MultiModalWorldModelConfig(vocab_size=10, img_channels=3, action_dim=4)
    cfg_s = MultiModalWorldModelConfig(vocab_size=10, img_channels=3, action_dim=4, embed_dim=64)
    teacher = MultiModalWorldModel(cfg_t)
    student = MultiModalWorldModel(cfg_s)

    # dummy dataset of random trajectories
    data = []
    for _ in range(8):
        t = torch.randint(0, 10, (1, 4))
        img = torch.randn(1, 3, 8, 8)
        a = torch.randint(0, 4, (1,))
        nt = torch.randint(0, 10, (1, 4))
        nimg = torch.randn(1, 3, 8, 8)
        r = torch.randn(())
        data.append((t, img, a, nt, nimg, r))
    dataset = TrajectoryDataset(data, lambda x: [int(c) for c in x[0]])

    cfg = DistillConfig(epochs=args.epochs, batch_size=2)
    distill_world_model(teacher, student, dataset, cfg)
    Path("student.pt").write_bytes(torch.save(student.state_dict(), Path("student.pt")) or b"")


if __name__ == "__main__":
    main()
