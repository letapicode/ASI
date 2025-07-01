import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn


@dataclass
class HeadPatch:
    layer: str
    head: int
    orig_q: torch.Tensor
    orig_k: torch.Tensor
    orig_v: torch.Tensor


class ActivationRecorder:
    """Record activations from specified modules via forward hooks."""

    def __init__(self, module: nn.Module, names: Iterable[str]) -> None:
        self.module = module
        self.names = list(names)
        self.records: Dict[str, torch.Tensor] = {}
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        for name in self.names:
            sub = dict(module.named_modules())[name]
            handle = sub.register_forward_hook(self._make_hook(name))
            self.handles.append(handle)

    def _make_hook(self, name: str):
        def hook(_mod: nn.Module, _inp: Tuple[torch.Tensor], out: torch.Tensor) -> None:
            with torch.no_grad():
                self.records[name] = out.detach()
        return hook

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


def record_attention_weights(model: nn.Module, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Run ``model`` on ``x`` and return attention weights keyed by module name."""
    weights: Dict[str, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []
    patches = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.MultiheadAttention):
            orig_forward = mod.forward

            def wrapped_forward(*args, _orig=orig_forward, **kwargs):
                kwargs["need_weights"] = True
                kwargs.setdefault("average_attn_weights", False)
                return _orig(*args, **kwargs)

            def hook(_mod: nn.MultiheadAttention, _inp, output, n=name):
                if isinstance(output, tuple) and len(output) == 2:
                    weights[n] = output[1].detach()

            mod.forward = wrapped_forward
            handles.append(mod.register_forward_hook(hook))
            patches.append((mod, orig_forward))
    model(x)
    for h in handles:
        h.remove()
    for mod, orig in patches:
        mod.forward = orig
    return weights


def _get_mha(model: nn.Module, name: str) -> nn.MultiheadAttention:
    mod = dict(model.named_modules()).get(name)
    if not isinstance(mod, nn.MultiheadAttention):
        raise ValueError(f"{name} is not MultiheadAttention")
    return mod


def zero_attention_head(model: nn.Module, name: str, head: int) -> HeadPatch:
    """Zero the projection weights of one attention head and return the patch."""
    attn = _get_mha(model, name)
    hdim = attn.head_dim
    start = head * hdim
    end = start + hdim
    patch = HeadPatch(
        layer=name,
        head=head,
        orig_q=attn.in_proj_weight[start:end].detach().clone(),
        orig_k=attn.in_proj_weight[attn.embed_dim + start : attn.embed_dim + end].detach().clone(),
        orig_v=attn.in_proj_weight[2 * attn.embed_dim + start : 2 * attn.embed_dim + end].detach().clone(),
    )
    with torch.no_grad():
        attn.in_proj_weight[start:end].zero_()
        attn.in_proj_weight[attn.embed_dim + start : attn.embed_dim + end].zero_()
        attn.in_proj_weight[2 * attn.embed_dim + start : 2 * attn.embed_dim + end].zero_()
    return patch


def restore_attention_head(model: nn.Module, patch: HeadPatch) -> None:
    """Restore a head previously zeroed with :func:`zero_attention_head`."""
    attn = _get_mha(model, patch.layer)
    hdim = attn.head_dim
    start = patch.head * hdim
    end = start + hdim
    with torch.no_grad():
        attn.in_proj_weight[start:end].copy_(patch.orig_q)
        attn.in_proj_weight[attn.embed_dim + start : attn.embed_dim + end].copy_(patch.orig_k)
        attn.in_proj_weight[2 * attn.embed_dim + start : 2 * attn.embed_dim + end].copy_(patch.orig_v)


@contextlib.contextmanager
def patched_head(model: nn.Module, name: str, head: int):
    """Context manager that zeros a head temporarily."""
    patch = zero_attention_head(model, name, head)
    try:
        yield
    finally:
        restore_attention_head(model, patch)


def head_importance(model: nn.Module, x: torch.Tensor, name: str) -> torch.Tensor:
    """Return L2 output differences when ablating each head."""
    attn = _get_mha(model, name)
    num_heads = attn.num_heads
    baseline = model(x).detach()
    diffs = []
    for h in range(num_heads):
        with patched_head(model, name, h):
            out = model(x).detach()
        diffs.append(torch.norm(baseline - out).item())
    return torch.tensor(diffs)


class AttentionVisualizer:
    """Save attention weights as heatmaps using existing hooks."""

    def __init__(self, model: nn.Module, names: Iterable[str] | None = None, out_dir: str = "attn_vis") -> None:
        if names is None:
            names = [n for n, m in model.named_modules() if isinstance(m, nn.MultiheadAttention)]
        self.model = model
        self.names = list(names)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _save_heatmap(self, attn: torch.Tensor, path: Path) -> None:
        arr = attn.detach().cpu().numpy()
        arr = arr - arr.min()
        arr = arr / (arr.max() + 1e-6)
        arr = (arr * 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")
        img = img.resize((attn.size(1) * 8, attn.size(0) * 8), Image.NEAREST)
        img.save(path)

    def run(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        weights = record_attention_weights(self.model, x)
        for name in self.names:
            w = weights.get(name)
            if w is None:
                continue
            avg = w.mean(dim=0)
            for h, hmap in enumerate(avg):
                fname = f"{name.replace('.', '_')}_h{h}.png"
                self._save_heatmap(hmap, self.out_dir / fname)
        return weights


__all__ = [
    "ActivationRecorder",
    "record_attention_weights",
    "zero_attention_head",
    "restore_attention_head",
    "patched_head",
    "head_importance",
    "AttentionVisualizer",
    "HeadPatch",
]
