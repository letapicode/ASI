from dataclasses import dataclass
from typing import Iterable, Tuple, Any, TYPE_CHECKING

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:  # pragma: no cover - for type hints
    from .hierarchical_memory import HierarchicalMemory
    from .sign_language import SignLanguageRecognizer


class TextEncoder(nn.Module):
    """Simple text encoder using an embedding table and transformer."""

    def __init__(self, vocab_size: int, dim: int, num_layers: int = 2) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=8)
        self.tr = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        x = self.tr(x)
        return x.mean(dim=1)


class ImageEncoder(nn.Module):
    """Basic CNN image encoder."""

    def __init__(self, in_channels: int, dim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, dim, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.conv(images)
        return x.view(x.size(0), -1)


class AudioEncoder(nn.Module):
    """1D convolutional encoder for audio spectrograms."""

    def __init__(self, in_channels: int, dim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, dim, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = self.conv(audio)
        return x.view(x.size(0), -1)


class FusionHead(nn.Module):
    """Projection head to map modality features into a shared space."""

    def __init__(self, dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.fc(x), dim=-1)


def _contrastive_loss(a: torch.Tensor, b: torch.Tensor, temperature: float) -> torch.Tensor:
    """Compute symmetric cross-entropy loss for embeddings ``a`` vs ``b``."""
    logits = a @ b.t() / temperature
    labels = torch.arange(a.size(0), device=a.device)
    loss_a = nn.functional.cross_entropy(logits, labels)
    loss_b = nn.functional.cross_entropy(logits.t(), labels)
    return (loss_a + loss_b) / 2


@dataclass
class CrossModalFusionConfig:
    vocab_size: int
    text_dim: int
    img_channels: int
    audio_channels: int
    latent_dim: int
    lr: float = 1e-4
    temperature: float = 0.07


class CrossModalFusion(nn.Module):
    """Unified encoder for text, images and audio."""

    def __init__(self, cfg: CrossModalFusionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.text_enc = TextEncoder(cfg.vocab_size, cfg.text_dim)
        self.img_enc = ImageEncoder(cfg.img_channels, cfg.text_dim)
        self.audio_enc = AudioEncoder(cfg.audio_channels, cfg.text_dim)
        self.text_proj = FusionHead(cfg.text_dim, cfg.latent_dim)
        self.img_proj = FusionHead(cfg.text_dim, cfg.latent_dim)
        self.audio_proj = FusionHead(cfg.text_dim, cfg.latent_dim)
        self.temperature = cfg.temperature

    def forward(
        self,
        text: torch.Tensor | None = None,
        images: torch.Tensor | None = None,
        audio: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        t_feat = self.text_proj(self.text_enc(text)) if text is not None else None
        i_feat = self.img_proj(self.img_enc(images)) if images is not None else None
        a_feat = self.audio_proj(self.audio_enc(audio)) if audio is not None else None
        return t_feat, i_feat, a_feat


class MultiModalDataset(Dataset):
    """Dataset of paired text, image, audio and optional sign video."""

    def __init__(self, triples: Iterable[Tuple[Any, Any, Any, Any | None]], tokenizer) -> None:
        self.items = list(triples)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        text, img, aud = item[:3]
        sign = item[3] if len(item) > 3 else None
        tokens = torch.tensor(self.tokenizer(text), dtype=torch.long)
        if sign is None:
            return tokens, img, aud
        return tokens, img, aud, sign


def train_fusion_model(
    model: CrossModalFusion,
    dataset: Dataset,
    epochs: int = 1,
    batch_size: int = 8,
) -> None:
    """Train model using CLIP-style contrastive loss."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=model.cfg.lr)
    device = next(model.parameters()).device
    model.train()
    for _ in range(epochs):
        for tokens, imgs, aud in loader:
            tokens, imgs, aud = tokens.to(device), imgs.to(device), aud.to(device)
            t_emb, i_emb, a_emb = model(tokens, imgs, aud)
            loss = _contrastive_loss(t_emb, i_emb, model.temperature)
            loss += _contrastive_loss(t_emb, a_emb, model.temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()


from .quantum_multimodal_retrieval import quantum_crossmodal_search


def encode_all(
    model: CrossModalFusion,
    dataset: Dataset,
    batch_size: int = 8,
    memory: "HierarchicalMemory | None" = None,
    *,
    quantum: bool = False,
    k: int = 5,
    sign_recognizer: "SignLanguageRecognizer | None" = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return all modality embeddings and optionally store them.

    When ``quantum`` is ``True`` and ``memory`` is provided, each fused
    embedding is queried using :func:`quantum_crossmodal_search` after being
    stored. The search results are not returned but this warms the quantum
    retrieval index.
    """
    loader = DataLoader(dataset, batch_size=batch_size)
    device = next(model.parameters()).device
    model.eval()
    text_vecs, img_vecs, aud_vecs, sign_vecs = [], [], [], []
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if len(batch) == 4:
                tokens, imgs, aud, sign = batch
            else:
                tokens, imgs, aud = batch
                sign = None
            tokens, imgs, aud = tokens.to(device), imgs.to(device), aud.to(device)
            t_emb, i_emb, a_emb = model(tokens, imgs, aud)
            text_vecs.append(t_emb.cpu())
            img_vecs.append(i_emb.cpu())
            aud_vecs.append(a_emb.cpu())
            s_emb = None
            if sign is not None and sign_recognizer is not None:
                s_list = [torch.from_numpy(sign_recognizer.encode(v)) for v in sign]
                s_emb = torch.stack(s_list)
                sign_vecs.append(s_emb)
            if memory is not None:
                start = idx * batch_size
                metas = [start + i for i in range(tokens.size(0))]
                memory.add_modalities(t_emb.cpu(), i_emb.cpu(), a_emb.cpu(), s_emb.cpu() if s_emb is not None else None, metas)
                memory.add_multimodal(t_emb.cpu(), i_emb.cpu(), a_emb.cpu(), s_emb.cpu() if s_emb is not None else None, metas)
                if quantum:
                    fused = (t_emb + i_emb + a_emb + (s_emb if s_emb is not None else 0)) / (4.0 if s_emb is not None else 3.0)
                    for q in fused:
                        quantum_crossmodal_search(q, memory, k=k)
    all_t = torch.cat(text_vecs, dim=0)
    all_i = torch.cat(img_vecs, dim=0)
    all_a = torch.cat(aud_vecs, dim=0)
    if sign_vecs:
        all_s = torch.cat(sign_vecs, dim=0)
        return all_t, all_i, all_a, all_s
    return all_t, all_i, all_a


def retrieval_accuracy(
    model: CrossModalFusion,
    dataset: Dataset,
    memory: "HierarchicalMemory",
    batch_size: int = 8,
    k: int = 1,
) -> float:
    """Return retrieval accuracy after encoding ``dataset`` into ``memory``."""

    t_vecs, i_vecs, a_vecs = encode_all(model, dataset, batch_size=batch_size, memory=memory)
    correct = 0
    for idx in range(len(dataset)):
        query = (t_vecs[idx] + i_vecs[idx] + a_vecs[idx]) / 3.0
        out, meta = memory.search(query, k=k)
        if meta and meta[0] == idx:
            correct += 1
    return correct / len(dataset)


__all__ = [
    "CrossModalFusionConfig",
    "CrossModalFusion",
    "MultiModalDataset",
    "train_fusion_model",
    "encode_all",
    "retrieval_accuracy",
]
