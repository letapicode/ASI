from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .cross_modal_fusion import CrossModalFusion

__all__ = ["MultiModalWorldModel", "train_world_model", "rollout"]


class MultiModalWorldModel(nn.Module):
    """A simple transformer-based world model for text, images and actions."""

    def __init__(self, vocab: int, action_dim: int, hidden_dim: int = 256, latent_dim: int = 128):
        super().__init__()
        self.fusion = CrossModalFusion(vocab, latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.action_embed = nn.Linear(action_dim, latent_dim)
        encoder_layer = nn.TransformerEncoderLayer(latent_dim, 4, hidden_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Linear(latent_dim, vocab)

    def forward(self, tokens: torch.Tensor, images: torch.Tensor, audio: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        t = self.fusion.encode_text(tokens)
        i = self.fusion.encode_image(images)
        a = self.fusion.encode_audio(audio)
        act = self.action_embed(actions)
        seq = torch.stack([t, i, a, act], dim=1)
        h = self.encoder(seq)
        out = self.head(h[:, 0])
        return out


def train_world_model(model: MultiModalWorldModel, data_loader, optim: torch.optim.Optimizer, device: str = "cpu"):
    model.train()
    for tokens, images, audio, actions, targets in data_loader:
        tokens = tokens.to(device)
        images = images.to(device)
        audio = audio.to(device)
        actions = actions.to(device)
        targets = targets.to(device)
        optim.zero_grad()
        logits = model(tokens, images, audio, actions)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optim.step()
        yield float(loss)


def rollout(model: MultiModalWorldModel, start_tokens: torch.Tensor, images: torch.Tensor, audio: torch.Tensor, actions: torch.Tensor, steps: int = 1) -> torch.Tensor:
    model.eval()
    tokens = start_tokens
    for _ in range(steps):
        logits = model(tokens, images, audio, actions)
        next_tok = torch.argmax(logits, dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_tok], dim=1)
    return tokens
