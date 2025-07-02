from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Any

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint


class ActionEncoder(nn.Module):
    """Embed discrete actions."""

    def __init__(self, action_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(action_dim, embed_dim)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.embed(a)


class ObservationEncoder(nn.Module):
    """Encode text and image observations."""

    def __init__(self, vocab_size: int, img_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.text_emb = nn.Embedding(vocab_size, embed_dim)
        self.img_conv = nn.Sequential(
            nn.Conv2d(img_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, embed_dim, 3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8)
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=2)

    def forward(self, text: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        t = self.text_emb(text).mean(dim=1)
        i = self.img_conv(image).flatten(1)
        merged = torch.stack([t, i], dim=1)
        return self.tr(merged).mean(dim=1)


class DynamicsModel(nn.Module):
    """Predict next latent state and reward."""

    def __init__(self, embed_dim: int, action_dim: int) -> None:
        super().__init__()
        dec_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=8)
        self.dec = nn.TransformerDecoder(dec_layer, num_layers=2)
        self.action_emb = nn.Embedding(action_dim, embed_dim)
        self.state_proj = nn.Linear(embed_dim, embed_dim)
        self.reward_head = nn.Linear(embed_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a = self.action_emb(action).unsqueeze(0)  # (1, embed_dim)
        h = self.dec(tgt=a.unsqueeze(0), memory=state.unsqueeze(0)).squeeze(0)
        next_state = self.state_proj(h)
        reward = self.reward_head(h).squeeze(-1)
        return next_state, reward


@dataclass
class MultiModalWorldModelConfig:
    vocab_size: int
    img_channels: int
    action_dim: int
    embed_dim: int = 128
    lr: float = 1e-4
    checkpoint_blocks: bool = False


class MultiModalWorldModel(nn.Module):
    """Unified world model over text, images and actions."""

    def __init__(self, cfg: MultiModalWorldModelConfig) -> None:
        super().__init__()
        self.obs_enc = ObservationEncoder(cfg.vocab_size, cfg.img_channels, cfg.embed_dim)
        self.dyn = DynamicsModel(cfg.embed_dim, cfg.action_dim)
        self.cfg = cfg

    def encode_obs(self, text: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        if self.cfg.checkpoint_blocks:
            return checkpoint(self.obs_enc, text, image)
        return self.obs_enc(text, image)

    def predict_dynamics(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.checkpoint_blocks:
            return checkpoint(self.dyn, state, action)
        return self.dyn(state, action)

    def forward(
        self,
        text: torch.Tensor,
        image: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        state = self.encode_obs(text, image)
        return self.predict_dynamics(state, action)


class TrajectoryDataset(Dataset):
    """(text, image, action, next_text, next_img, reward) tuples."""

    def __init__(self, entries: Iterable[Tuple[Any, Any, Any, Any, Any, float]], tokenizer) -> None:
        self.data = list(entries)
        self.tokenizer = tokenizer

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.data)

    def __getitem__(self, idx: int):  # type: ignore[override]
        t, img, a, nt, nimg, r = self.data[idx]
        t_tk = torch.tensor(self.tokenizer(t), dtype=torch.long)
        nt_tk = torch.tensor(self.tokenizer(nt), dtype=torch.long)
        return t_tk, img, a, nt_tk, nimg, torch.tensor(r, dtype=torch.float32)


def train_world_model(
    model: MultiModalWorldModel,
    dataset: Dataset,
    epochs: int = 1,
    batch_size: int = 8,
) -> None:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=model.cfg.lr)
    device = next(model.parameters()).device
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for t, img, a, nt, nimg, r in loader:
            t, img, a, nt, nimg, r = (
                t.to(device),
                img.to(device),
                a.to(device),
                nt.to(device),
                nimg.to(device),
                r.to(device),
            )
            state = model.encode_obs(t, img)
            target = model.encode_obs(nt, nimg)
            pred_state, pred_reward = model.predict_dynamics(state, a)
            loss = loss_fn(pred_state, target) + loss_fn(pred_reward, r)
            opt.zero_grad()
            loss.backward()
            opt.step()


def rollout(
    model: MultiModalWorldModel,
    start_text: torch.Tensor,
    start_img: torch.Tensor,
    policy_fn,
    steps: int = 10,
) -> Tuple[list[torch.Tensor], list[float]]:
    device = next(model.parameters()).device
    text = start_text.to(device)
    img = start_img.to(device)
    states = []
    rewards = []
    with torch.no_grad():
        for _ in range(steps):
            state = model.encode_obs(text, img)
            action = policy_fn(state)
            next_state, reward = model.predict_dynamics(state, action)
            states.append(next_state.cpu())
            rewards.append(float(reward.item()))
            text = text  # placeholder for decoded update
            img = img
    return states, rewards


__all__ = [
    "MultiModalWorldModelConfig",
    "MultiModalWorldModel",
    "TrajectoryDataset",
    "train_world_model",
    "rollout",
]
