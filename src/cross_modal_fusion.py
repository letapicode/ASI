import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["CrossModalFusion", "contrastive_loss", "train_fusion_step"]


class CrossModalFusion(nn.Module):
    """Embed text, images and audio into a shared latent space."""

    def __init__(self, vocab: int, latent_dim: int = 256, hidden_dim: int = 512, image_channels: int = 3, audio_dim: int = 80):
        super().__init__()
        self.text_embed = nn.Embedding(vocab, hidden_dim)
        self.text_proj = nn.Linear(hidden_dim, latent_dim)

        self.image_conv = nn.Sequential(
            nn.Conv2d(image_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.image_proj = nn.Linear(hidden_dim, latent_dim)

        self.audio_proj = nn.Linear(audio_dim, latent_dim)
        self.fuse = nn.Linear(latent_dim, latent_dim)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.text_embed(tokens).mean(dim=1)
        return F.normalize(self.fuse(self.text_proj(h)), dim=-1)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        h = self.image_conv(images).squeeze(-1).squeeze(-1)
        return F.normalize(self.fuse(self.image_proj(h)), dim=-1)

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        h = audio.mean(dim=-1)
        return F.normalize(self.fuse(self.audio_proj(h)), dim=-1)


def contrastive_loss(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Compute symmetric contrastive loss between two embedding sets."""
    logits = a @ b.t() / temperature
    labels = torch.arange(len(a), device=a.device)
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
    return loss


def train_fusion_step(model: CrossModalFusion, text: torch.Tensor, images: torch.Tensor, audio: torch.Tensor, optim: torch.optim.Optimizer) -> float:
    model.train()
    optim.zero_grad()
    t = model.encode_text(text)
    i = model.encode_image(images)
    a = model.encode_audio(audio)
    loss = contrastive_loss(t, i) + contrastive_loss(t, a) + contrastive_loss(i, a)
    loss.backward()
    optim.step()
    return float(loss)
