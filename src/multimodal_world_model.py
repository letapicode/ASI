import torch
from torch import nn
from torch.nn import functional as F
from typing import Iterable, Tuple, List


class MultiModalWorldModel(nn.Module):
    """Unified transformer for text, image and low-level actions."""

    def __init__(
        self,
        vocab_size: int = 1000,
        image_dim: int = 64,
        action_dim: int = 32,
        hidden: int = 256,
        num_layers: int = 2,
        nhead: int = 4,
    ) -> None:
        super().__init__()
        self.text_embed = nn.Embedding(vocab_size, hidden)
        self.image_proj = nn.Linear(image_dim, hidden)
        self.action_embed = nn.Embedding(action_dim, hidden)
        self.pos = nn.Parameter(torch.randn(512, hidden))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out = nn.Linear(hidden, hidden)

    def forward(
        self,
        text_tokens: torch.Tensor,
        image_feats: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Return latent next-state predictions."""
        batch, seq = text_tokens.shape
        t = self.text_embed(text_tokens)
        i = self.image_proj(image_feats)
        a = self.action_embed(actions)
        x = t + i + a + self.pos[:seq]
        h = self.transformer(x)
        return self.out(h)

    def predict_next(
        self,
        text_tokens: torch.Tensor,
        image_feats: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(text_tokens, image_feats, actions)


def train_world_model(
    model: MultiModalWorldModel,
    dataset: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    epochs: int = 1,
    lr: float = 1e-3,
) -> None:
    """Train ``model`` on ``dataset`` of (text, image, action, target) tuples."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for text, image, action, target in dataset:
            pred = model(text, image, action)
            loss = F.mse_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()


def rollout(
    model: MultiModalWorldModel,
    start: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    steps: int = 5,
) -> List[torch.Tensor]:
    """Generate a sequence of latent states."""
    model.eval()
    text, image, action = start
    outs: List[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(steps):
            out = model.predict_next(text, image, action)
            outs.append(out)
            text = out.argmax(-1)
            image = out
            action = out.argmax(-1)
    return outs
