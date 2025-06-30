import torch
from torch import nn
from torch.nn import functional as F
from typing import Iterable, Tuple

from .cross_modal_fusion import CrossModalFusion, train_fusion


class SkillTransferModel(nn.Module):
    """Map video demonstrations to robot policies."""

    def __init__(self, action_dim: int = 32, hidden: int = 256) -> None:
        super().__init__()
        self.fusion = CrossModalFusion(hidden=hidden)
        self.policy = nn.Linear(hidden, action_dim)

    def forward(self, text, image, audio) -> torch.Tensor:
        rep = self.fusion(text, image, audio)
        return self.policy(rep)


def transfer_skills(
    pretrain: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    finetune: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    action_dim: int = 32,
    epochs: int = 1,
    lr: float = 1e-3,
) -> SkillTransferModel:
    """Pre-train on demonstrations then fine-tune on real robot samples."""
    model = SkillTransferModel(action_dim=action_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # Pretrain fusion module
    fusion_data = ((t, i, a) for t, i, a, _ in pretrain)
    train_fusion(model.fusion, fusion_data, epochs=epochs, lr=lr)
    # Supervised policy learning
    for _ in range(epochs):
        for text, image, audio, action in finetune:
            pred = model(text, image, audio)
            loss = F.cross_entropy(pred.view(-1, action_dim), action.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model
