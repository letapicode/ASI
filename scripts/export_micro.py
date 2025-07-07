#!/usr/bin/env python
"""Export models to microcontroller-friendly formats."""

from pathlib import Path

from src.multimodal_world_model import MultiModalWorldModel, MultiModalWorldModelConfig
from src.cross_modal_fusion import CrossModalFusion, CrossModalFusionConfig
from src.onnx_utils import export_to_onnx
from src.micro_export import export_to_tflite_micro, export_to_microtvm


def main(out_dir: str = "micro_models") -> None:
    out = Path(out_dir)
    out.mkdir(exist_ok=True)

    wm_cfg = MultiModalWorldModelConfig(vocab_size=10, img_channels=3, action_dim=2, embed_dim=8)
    fusion_cfg = CrossModalFusionConfig(
        vocab_size=10, text_dim=8, img_channels=3, audio_channels=1, latent_dim=4
    )

    wm = MultiModalWorldModel(wm_cfg)
    fusion = CrossModalFusion(fusion_cfg)

    wm_onnx = out / "world_model.onnx"
    fusion_onnx = out / "fusion.onnx"
    export_to_onnx(wm, str(wm_onnx))
    export_to_onnx(fusion, str(fusion_onnx))

    export_to_tflite_micro(wm_onnx, out / "world_model.tflite")
    export_to_tflite_micro(fusion_onnx, out / "fusion.tflite")

    export_to_microtvm(wm_onnx, out / "world_model_micro.tar")
    export_to_microtvm(fusion_onnx, out / "fusion_micro.tar")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export models for micro inference")
    parser.add_argument("--out-dir", default="micro_models", help="Output directory")
    args = parser.parse_args()
    main(args.out_dir)

