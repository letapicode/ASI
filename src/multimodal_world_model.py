from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Any

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
from .spiking_layers import SpikingLinear
from .telemetry import TelemetryLogger
try:
    from .lora_quant import apply_quant_lora
except Exception:  # pragma: no cover - fallback for tests
    try:
        from asi.lora_quant import apply_quant_lora  # type: ignore
    except Exception:
        import importlib.util
        import sys
        from pathlib import Path

        spec = importlib.util.spec_from_file_location(
            "lora_quant", Path(__file__).with_name("lora_quant.py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["lora_quant"] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)  # type: ignore
        from lora_quant import apply_quant_lora  # type: ignore

try:
    from .fpga_backend import FPGAAccelerator, _HAS_FPGA
except Exception:  # pragma: no cover - fallback for tests
    FPGAAccelerator = None  # type: ignore
    _HAS_FPGA = False

try:
    from .analog_backend import AnalogAccelerator, _HAS_ANALOG
except Exception:  # pragma: no cover - fallback for tests
    AnalogAccelerator = None  # type: ignore
    _HAS_ANALOG = False


def _replace_mlps(mod: nn.Module, *, use_loihi: bool = False) -> None:
    """Recursively swap MLP ``Linear`` layers with ``SpikingLinear``."""
    for name, child in list(mod.named_children()):
        if isinstance(child, nn.Linear) and name in {"linear1", "linear2", "state_proj", "reward_head"}:
            new = SpikingLinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                use_loihi=use_loihi,
            )
            with torch.no_grad():
                new.linear.weight.copy_(child.weight)
                if child.bias is not None:
                    assert new.linear.bias is not None
                    new.linear.bias.copy_(child.bias)
            setattr(mod, name, new)
        else:
            _replace_mlps(child, use_loihi=use_loihi)


class ActionEncoder(nn.Module):
    """Embed discrete actions."""

    def __init__(self, action_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(action_dim, embed_dim)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.embed(a)


class ObservationEncoder(nn.Module):
    """Encode text and image observations."""

    def __init__(
        self,
        vocab_size: int,
        img_channels: int,
        embed_dim: int,
        use_event_streams: bool = False,
        event_channels: int = 0,
    ) -> None:
        super().__init__()
        self.text_emb = nn.Embedding(vocab_size, embed_dim)
        self.img_conv = nn.Sequential(
            nn.Conv2d(img_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, embed_dim, 3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.event_enc: nn.Module | None = None
        if use_event_streams and event_channels > 0:
            self.event_enc = nn.Sequential(
                nn.Conv1d(event_channels, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, embed_dim, 3, padding=1),
                nn.AdaptiveAvgPool1d(1),
            )
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8)
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=2)

    def forward(
        self,
        text: torch.Tensor,
        image: torch.Tensor,
        events: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t = self.text_emb(text).mean(dim=1)
        i = self.img_conv(image).flatten(1)
        parts = [t, i]
        if self.event_enc is not None and events is not None:
            if events.dim() == 2:
                events = events.unsqueeze(0)
            e = self.event_enc(events).squeeze(-1)
            if e.dim() == 2:
                parts.append(e)
            else:
                parts.append(e.unsqueeze(0))
        merged = torch.stack(parts, dim=1)
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
        a = self.action_emb(action).unsqueeze(0)  # (1, B, E)
        h = self.dec(tgt=a, memory=state.unsqueeze(0)).squeeze(0)
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
    use_lora: bool = False
    use_spiking: bool = False
    use_loihi: bool = False
    use_fpga: bool = False
    use_analog: bool = False
    use_event_streams: bool = False
    event_channels: int = 0


class MultiModalWorldModel(nn.Module):
    """Unified world model over text, images and actions."""

    def __init__(self, cfg: MultiModalWorldModelConfig) -> None:
        super().__init__()
        self.obs_enc = ObservationEncoder(
            cfg.vocab_size,
            cfg.img_channels,
            cfg.embed_dim,
            use_event_streams=cfg.use_event_streams,
            event_channels=cfg.event_channels,
        )
        self.dyn = DynamicsModel(cfg.embed_dim, cfg.action_dim)
        self.cfg = cfg
        if cfg.use_lora:
            apply_quant_lora(
                self.obs_enc,
                ["linear1", "linear2", "out_proj", "q_proj", "k_proj", "v_proj"],
            )
            apply_quant_lora(
                self.dyn,
                [
                    "linear1",
                    "linear2",
                    "out_proj",
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "state_proj",
                    "reward_head",
                ],
            )
        if cfg.use_spiking:
            _replace_mlps(self.obs_enc, use_loihi=cfg.use_loihi)
            _replace_mlps(self.dyn, use_loihi=cfg.use_loihi)
        self.fpga: FPGAAccelerator | None = None
        if cfg.use_fpga and FPGAAccelerator is not None and _HAS_FPGA:
            self.fpga = FPGAAccelerator(self, forward_fn=self._forward_impl)
            self.fpga.compile()
        self.analog: AnalogAccelerator | None = None
        if cfg.use_analog and AnalogAccelerator is not None and _HAS_ANALOG:
            self.analog = AnalogAccelerator()

    def encode_obs(
        self,
        text: torch.Tensor,
        image: torch.Tensor,
        events: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.cfg.checkpoint_blocks:
            return checkpoint(self.obs_enc, text, image, events)
        return self.obs_enc(text, image, events)

    def predict_dynamics(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.checkpoint_blocks:
            return checkpoint(self.dyn, state, action)
        return self.dyn(state, action)

    def _forward_impl(
        self,
        text: torch.Tensor,
        image: torch.Tensor,
        action: torch.Tensor,
        events: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        state = self.encode_obs(text, image, events)
        return self.predict_dynamics(state, action)

    def forward(
        self,
        text: torch.Tensor,
        image: torch.Tensor,
        action: torch.Tensor,
        events: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.fpga is not None:
            return self.fpga.run(text, image, action, events)
        if self.analog is not None:
            with self.analog:
                return self._forward_impl(text, image, action, events)
        return self._forward_impl(text, image, action, events)


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
    event_dataset: Dataset | None = None,
    telemetry: "TelemetryLogger | None" = None,
    epochs: int = 1,
    batch_size: int = 8,
) -> None:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    event_iter = None
    if event_dataset is not None:
        from itertools import cycle

        event_loader = DataLoader(event_dataset, batch_size=batch_size, shuffle=True)
        event_iter = cycle(event_loader)
    opt = torch.optim.Adam(model.parameters(), lr=model.cfg.lr)
    device = next(model.parameters()).device
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        total = 0.0
        for t, img, a, nt, nimg, r in loader:
            t, img, a, nt, nimg, r = (
                t.to(device),
                img.to(device),
                a.to(device),
                nt.to(device),
                nimg.to(device),
                r.to(device),
            )
            events = next(event_iter) if event_iter is not None else None
            if events is not None:
                events = events.to(device)
            state = model.encode_obs(t, img, events)
            target = model.encode_obs(nt, nimg, events)
            pred_state, pred_reward = model.predict_dynamics(state, a)
            loss = loss_fn(pred_state, target) + loss_fn(pred_reward, r)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        if telemetry is not None:
            telemetry.metrics["world_model_loss"] = total / len(loader)


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
