import argparse
import torch
from asi.lora_merger import merge_adapters


def main(out_path: str, adapters: list[str], weights: list[float]) -> None:
    merged = merge_adapters(None, adapters, weights)
    torch.save(merged, out_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Merge LoRA checkpoints")
    p.add_argument("--out", required=True)
    p.add_argument("--adapters", nargs="+", required=True)
    p.add_argument("--weights", nargs="*", type=float)
    args = p.parse_args()
    w = args.weights if args.weights else [1.0] * len(args.adapters)
    main(args.out, args.adapters, w)
