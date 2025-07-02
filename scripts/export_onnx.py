import argparse
import os

from src.multimodal_world_model import MultiModalWorldModel, MultiModalWorldModelConfig
from src.cross_modal_fusion import CrossModalFusion, CrossModalFusionConfig
from src.onnx_utils import export_to_onnx


def main(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    wm_cfg = MultiModalWorldModelConfig(vocab_size=10, img_channels=3, action_dim=2, embed_dim=8)
    wm = MultiModalWorldModel(wm_cfg)
    export_to_onnx(wm, os.path.join(out_dir, "multimodal_world_model.onnx"))

    fusion_cfg = CrossModalFusionConfig(
        vocab_size=10,
        text_dim=8,
        img_channels=3,
        audio_channels=1,
        latent_dim=4,
    )
    fusion = CrossModalFusion(fusion_cfg)
    export_to_onnx(fusion, os.path.join(out_dir, "cross_modal_fusion.onnx"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export models to ONNX format")
    parser.add_argument("--out-dir", default=".", help="Output directory")
    args = parser.parse_args()
    main(args.out_dir)
