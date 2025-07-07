from dataclasses import dataclass
from typing import Iterable, Tuple, Any, TYPE_CHECKING
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:  # pragma: no cover - for type hints
    from .hierarchical_memory import HierarchicalMemory


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


class BCIEncoder(nn.Module):
    """1D convolutional encoder for EEG/ECoG signals."""

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

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        x = self.conv(signals)
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
    bci_channels: int = 0
    lr: float = 1e-4
    temperature: float = 0.07


class CrossModalFusion(nn.Module):
    """Unified encoder for text, images, audio and BCI signals."""

    def __init__(self, cfg: CrossModalFusionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.text_enc = TextEncoder(cfg.vocab_size, cfg.text_dim)
        self.img_enc = ImageEncoder(cfg.img_channels, cfg.text_dim)
        self.audio_enc = AudioEncoder(cfg.audio_channels, cfg.text_dim)
        self.bci_enc = (
            BCIEncoder(cfg.bci_channels, cfg.text_dim) if cfg.bci_channels > 0 else None
        )
        self.text_proj = FusionHead(cfg.text_dim, cfg.latent_dim)
        self.img_proj = FusionHead(cfg.text_dim, cfg.latent_dim)
        self.audio_proj = FusionHead(cfg.text_dim, cfg.latent_dim)
        self.bci_proj = (
            FusionHead(cfg.text_dim, cfg.latent_dim) if cfg.bci_channels > 0 else None
        )
        self.temperature = cfg.temperature

    def forward(
        self,
        text: torch.Tensor | None = None,
        images: torch.Tensor | None = None,
        audio: torch.Tensor | None = None,
        bci: torch.Tensor | None = None,
    ) -> Tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        t_feat = self.text_proj(self.text_enc(text)) if text is not None else None
        i_feat = self.img_proj(self.img_enc(images)) if images is not None else None
        a_feat = self.audio_proj(self.audio_enc(audio)) if audio is not None else None
        b_feat = (
            self.bci_proj(self.bci_enc(bci))
            if (bci is not None and self.bci_enc is not None)
            else None
        )
        return t_feat, i_feat, a_feat, b_feat


class MultiModalDataset(Dataset):
    """Dataset of paired text, image, audio and optional BCI signals."""

    def __init__(
        self,
        entries: Iterable[Tuple[Any, Any, Any, Any | None]],
        tokenizer,
        bci_shape: Tuple[int, int] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.items = []
        bci_shape = bci_shape or (1, 8)
        for e in entries:
            if len(e) == 3:
                text, img, aud = e
                bci = torch.zeros(bci_shape, dtype=torch.float32)
            else:
                text, img, aud, bci = e
                bci = torch.as_tensor(bci, dtype=torch.float32)
            self.items.append((text, img, aud, bci))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        text, img, aud, bci = self.items[idx]
        tokens = torch.tensor(self.tokenizer(text), dtype=torch.long)
        return tokens, img, aud, bci


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
        for tokens, imgs, aud, bci in loader:
            tokens, imgs, aud, bci = (
                tokens.to(device),
                imgs.to(device),
                aud.to(device),
                bci.to(device),
            )
            t_emb, i_emb, a_emb, b_emb = model(tokens, imgs, aud, bci)
            loss = _contrastive_loss(t_emb, i_emb, model.temperature)
            loss += _contrastive_loss(t_emb, a_emb, model.temperature)
            if b_emb is not None:
                loss += _contrastive_loss(t_emb, b_emb, model.temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()


from .quantum_multimodal_retrieval import quantum_crossmodal_search
from .sign_language import SignLanguageRecognizer


def encode_all(
    model: CrossModalFusion,
    dataset: Dataset,
    batch_size: int = 8,
    memory: "HierarchicalMemory | None" = None,
    *,
    quantum: bool = False,
    k: int = 5,
    include_bci: bool = False,
    sign_videos: Iterable[Any] | None = None,
    include_sign: bool = False,
    recognizer: SignLanguageRecognizer | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return all modality embeddings and optionally store them.

    When ``quantum`` is ``True`` and ``memory`` is provided, each fused
    embedding is queried using :func:`quantum_crossmodal_search` after being
    stored. The search results are not returned but this warms the quantum
    retrieval index.
    """
    loader = DataLoader(dataset, batch_size=batch_size)
    device = next(model.parameters()).device
    model.eval()
    text_vecs, img_vecs, aud_vecs, bci_vecs, sign_vecs = [], [], [], [], []
    recognizer = recognizer or SignLanguageRecognizer()
    with torch.no_grad():
        for idx, (tokens, imgs, aud, bci) in enumerate(loader):
            tokens, imgs, aud, bci = (
                tokens.to(device),
                imgs.to(device),
                aud.to(device),
                bci.to(device),
            )
            t_emb, i_emb, a_emb, b_emb = model(tokens, imgs, aud, bci)
            sign_batch = None
            if include_sign and sign_videos is not None:
                start = idx * batch_size
                vids = sign_videos[start : start + tokens.size(0)]
                embeds = []
                for v in vids:
                    txt = recognizer.recognize(np.asarray(v))
                    tok = torch.tensor(dataset.tokenizer(txt), dtype=torch.long, device=device)
                    emb = model.text_proj(model.text_enc(tok.unsqueeze(0)))[0]
                    embeds.append(emb)
                sign_batch = torch.stack(embeds)
                sign_vecs.append(sign_batch.cpu())
            text_vecs.append(t_emb.cpu())
            img_vecs.append(i_emb.cpu())
            aud_vecs.append(a_emb.cpu())
            bci_vecs.append(b_emb.cpu())
            if memory is not None:
                start = idx * batch_size
                metas = [start + i for i in range(tokens.size(0))]
                memory.add_multimodal(
                    t_emb.cpu(), i_emb.cpu(), a_emb.cpu(), b_emb.cpu(), sign_batch.cpu() if sign_batch is not None else None, metas
                )
                if quantum:
                    vec_list = [t_emb, i_emb, a_emb]
                    if b_emb is not None:
                        vec_list.append(b_emb)
                    if sign_batch is not None:
                        vec_list.append(sign_batch)
                    fused = sum(vec_list) / len(vec_list)
                    for q in fused:
                        quantum_crossmodal_search(q, memory, k=k)
    all_t = torch.cat(text_vecs, dim=0)
    all_i = torch.cat(img_vecs, dim=0)
    all_a = torch.cat(aud_vecs, dim=0)
    if include_bci and include_sign:
        all_b = torch.cat(bci_vecs, dim=0)
        all_s = torch.cat(sign_vecs, dim=0)
        return all_t, all_i, all_a, all_b, all_s
    if include_bci:
        all_b = torch.cat(bci_vecs, dim=0)
        return all_t, all_i, all_a, all_b
    if include_sign:
        all_s = torch.cat(sign_vecs, dim=0)
        return all_t, all_i, all_a, all_s
    return all_t, all_i, all_a, None


def retrieval_accuracy(
    model: CrossModalFusion,
    dataset: Dataset,
    memory: "HierarchicalMemory",
    batch_size: int = 8,
    k: int = 1,
    include_bci: bool = False,
    include_sign: bool = False,
) -> float:
    """Return retrieval accuracy after encoding ``dataset`` into ``memory``."""

    out = encode_all(
        model,
        dataset,
        batch_size=batch_size,
        memory=memory,
        include_bci=include_bci,
        include_sign=include_sign,
    )
    if include_bci and include_sign:
        t_vecs, i_vecs, a_vecs, b_vecs, s_vecs = out
    elif include_bci:
        t_vecs, i_vecs, a_vecs, b_vecs = out
        s_vecs = None
    elif include_sign:
        t_vecs, i_vecs, a_vecs, s_vecs = out
        b_vecs = None
    else:
        t_vecs, i_vecs, a_vecs, _ = out
        b_vecs = s_vecs = None
    correct = 0
    for idx in range(len(dataset)):
        if include_bci and include_sign:
            query = (t_vecs[idx] + i_vecs[idx] + a_vecs[idx] + b_vecs[idx] + s_vecs[idx]) / 5.0
        elif include_bci:
            query = (t_vecs[idx] + i_vecs[idx] + a_vecs[idx] + b_vecs[idx]) / 4.0
        elif include_sign:
            query = (t_vecs[idx] + i_vecs[idx] + a_vecs[idx] + s_vecs[idx]) / 4.0
        else:
            query = (t_vecs[idx] + i_vecs[idx] + a_vecs[idx]) / 3.0
        out, meta = memory.search(query, k=k)
        if meta and meta[0] == idx:
            correct += 1
    return correct / len(dataset)


__all__ = [
    "CrossModalFusionConfig",
    "CrossModalFusion",
    "BCIEncoder",
    "MultiModalDataset",
    "train_fusion_model",
    "encode_all",
    "retrieval_accuracy",
]
