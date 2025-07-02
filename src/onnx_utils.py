from __future__ import annotations

import torch
from torch import nn

from .multimodal_world_model import MultiModalWorldModel
from .cross_modal_fusion import CrossModalFusion


def export_to_onnx(model: nn.Module, path: str) -> None:
    """Export supported models to ONNX format."""
    model.eval()
    with torch.no_grad():
        if isinstance(model, MultiModalWorldModel):
            cfg = model.cfg
            example = (
                torch.randint(0, cfg.vocab_size, (1, 4)),
                torch.randn(1, cfg.img_channels, 8, 8),
                torch.randint(0, cfg.action_dim, (1,)),
            )
            input_names = ["text", "image", "action"]
            output_names = ["state", "reward"]
        elif isinstance(model, CrossModalFusion):
            cfg = model.cfg
            example = (
                torch.randint(0, cfg.vocab_size, (1, 5)),
                torch.randn(1, cfg.img_channels, 16, 16),
                torch.randn(1, cfg.audio_channels, 32),
            )
            input_names = ["text", "images", "audio"]
            output_names = ["text_feat", "img_feat", "audio_feat"]
        else:
            raise TypeError("Unsupported model type for ONNX export")

        torch.onnx.export(
            model,
            example,
            path,
            input_names=input_names,
            output_names=output_names,
            opset_version=11,
        )


__all__ = ["export_to_onnx"]
