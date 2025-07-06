#!/usr/bin/env python
"""Export models to WebAssembly format using onnxruntime-web."""

from pathlib import Path

from asi.multimodal_world_model import MultiModalWorldModel, MultiModalWorldModelConfig
from asi.cross_modal_fusion import CrossModalFusion, CrossModalFusionConfig
from asi.onnx_utils import export_to_onnx, export_to_wasm


def main() -> None:
    out_dir = Path("wasm_models")
    out_dir.mkdir(exist_ok=True)

    world_cfg = MultiModalWorldModelConfig(vocab_size=10, img_channels=3, action_dim=4)
    fusion_cfg = CrossModalFusionConfig(
        vocab_size=10, text_dim=32, img_channels=3, audio_channels=2, latent_dim=16
    )

    world_model = MultiModalWorldModel(world_cfg)
    fusion_model = CrossModalFusion(fusion_cfg)

    world_onnx = out_dir / "world_model.onnx"
    fusion_onnx = out_dir / "fusion.onnx"

    export_to_onnx(world_model, str(world_onnx))
    export_to_onnx(fusion_model, str(fusion_onnx))

    export_to_wasm(world_onnx, out_dir / "world_model")
    export_to_wasm(fusion_onnx, out_dir / "fusion")


if __name__ == "__main__":
    main()
