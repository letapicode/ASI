import torch
from torch import nn
from torch.nn import functional as F
from typing import Iterable, Tuple, List


class CrossModalFusion(nn.Module):
    """Embed text, image and audio into a shared latent space."""

    def __init__(
        self,
        vocab_size: int = 1000,
        audio_dim: int = 80,
        image_dim: int = 64,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.text_embed = nn.Embedding(vocab_size, hidden)
        self.audio_proj = nn.Linear(audio_dim, hidden)
        self.image_proj = nn.Linear(image_dim, hidden)
        self.fuser = nn.Linear(hidden * 3, hidden)

    def forward(
        self,
        text: torch.Tensor,
        image: torch.Tensor,
        audio: torch.Tensor,
    ) -> torch.Tensor:
        """Return fused representation."""
        t = self.text_embed(text)
        i = self.image_proj(image)
        a = self.audio_proj(audio)
        h = torch.cat([t, i, a], dim=-1)
        return F.relu(self.fuser(h))

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        return self.text_embed(text)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.image_proj(image)

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        return self.audio_proj(audio)

    def similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        return (x * y).sum(-1)


def train_fusion(
    model: CrossModalFusion,
    dataset: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    epochs: int = 1,
    lr: float = 1e-3,
) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for text, image, audio in dataset:
            rep = model(text, image, audio)
            loss = (1 - F.cosine_similarity(rep[:-1], rep[1:]).mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
