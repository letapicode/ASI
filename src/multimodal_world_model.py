import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Iterable, Tuple


class MultiModalWorldModel(nn.Module):
    """Unified transformer for text, image and action dynamics."""

    def __init__(
        self,
        vocab_size: int,
        action_size: int,
        image_channels: int = 3,
        dim: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.text_emb = nn.Embedding(vocab_size, dim)
        self.action_emb = nn.Embedding(action_size, dim)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(image_channels, dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        encoder_layer = nn.TransformerEncoderLayer(dim, n_heads, dim * 4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.head = nn.Linear(dim, dim)
        self.register_parameter("pos", nn.Parameter(torch.randn(3, dim)))

    def forward(
        self, text: torch.Tensor, image: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Predict the next state vector.

        Args:
            text: Token ids ``(batch, seq)``.
            image: Image tensor ``(batch, C, H, W)``.
            action: Action ids ``(batch,)``.
        Returns:
            Tensor ``(batch, dim)`` representing the next state.
        """
        t_emb = self.text_emb(text).mean(dim=1)
        i_emb = self.image_encoder(image)
        a_emb = self.action_emb(action)
        tokens = torch.stack([t_emb, i_emb, a_emb], dim=1) + self.pos
        hidden = self.transformer(tokens)
        return self.head(hidden.mean(dim=1))


def train_world_model(
    model: MultiModalWorldModel,
    loader: DataLoader,
    epochs: int = 1,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
) -> MultiModalWorldModel:
    """Train the world model on the provided data loader."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        for text, image, action, target in loader:
            text = text.to(device)
            image = image.to(device)
            action = action.to(device)
            target = target.to(device)
            opt.zero_grad()
            pred = model(text, image, action)
            loss = loss_fn(pred, target)
            loss.backward()
            opt.step()
    return model


def rollout(
    model: MultiModalWorldModel,
    texts: torch.Tensor,
    images: torch.Tensor,
    actions: torch.Tensor,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Generate predicted states for a sequence of inputs."""
    model.eval()
    model.to(device)
    seq_len = texts.size(1)
    preds = []
    with torch.no_grad():
        for t in range(seq_len):
            pred = model(
                texts[:, t].to(device),
                images[:, t].to(device),
                actions[:, t].to(device),
            )
            preds.append(pred)
    return torch.stack(preds, dim=1)
