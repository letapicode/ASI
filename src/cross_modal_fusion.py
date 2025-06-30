import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Iterable, Dict


class TextEncoder(nn.Module):
    """Simple Transformer-based text encoder."""

    def __init__(self, vocab_size: int, dim: int, hidden: int = 256, layers: int = 2) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        block = nn.TransformerEncoderLayer(dim, 4, hidden, batch_first=True)
        self.encoder = nn.TransformerEncoder(block, layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        return self.encoder(x).mean(dim=1)


class ImageEncoder(nn.Module):
    """Lightweight CNN image encoder."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.conv(images)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class AudioEncoder(nn.Module):
    """1D convolutional audio encoder."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, dim)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = self.conv(audio)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class CrossModalFusionModel(nn.Module):
    """Embed text, images and audio in a shared latent space."""

    def __init__(self, vocab_size: int, dim: int = 128) -> None:
        super().__init__()
        self.text_encoder = TextEncoder(vocab_size, dim)
        self.image_encoder = ImageEncoder(dim)
        self.audio_encoder = AudioEncoder(dim)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        text: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        reps: Dict[str, torch.Tensor] = {}
        if text is not None:
            reps["text"] = F.normalize(self.text_encoder(text), dim=-1)
        if images is not None:
            reps["image"] = F.normalize(self.image_encoder(images), dim=-1)
        if audio is not None:
            reps["audio"] = F.normalize(self.audio_encoder(audio), dim=-1)
        return reps


def contrastive_loss(x: torch.Tensor, y: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    """Symmetric cross-entropy loss between two batches of embeddings."""
    scale = logit_scale.exp()
    logits = scale * x @ y.t()
    t = torch.arange(x.size(0), device=x.device)
    loss_a = F.cross_entropy(logits, t)
    loss_b = F.cross_entropy(logits.t(), t)
    return (loss_a + loss_b) / 2


def train_fusion_model(
    dataloader: Iterable[Dict[str, torch.Tensor]],
    model: CrossModalFusionModel,
    optimizer: torch.optim.Optimizer,
    steps: int,
    device: Optional[torch.device] = None,
) -> None:
    """Train ``model`` for a number of ``steps`` using contrastive losses."""
    device = device or next(model.parameters()).device
    model.train()
    for i, batch in enumerate(dataloader):
        if i >= steps:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        outs = model(**batch)
        loss = torch.tensor(0.0, device=device)
        if "text" in outs and "image" in outs:
            loss = loss + contrastive_loss(outs["text"], outs["image"], model.logit_scale)
        if "text" in outs and "audio" in outs:
            loss = loss + contrastive_loss(outs["text"], outs["audio"], model.logit_scale)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

