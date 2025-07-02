import argparse
from pathlib import Path
import torch
from asi.transformer_circuits import AttentionVisualizer


def simple_tokenize(text: str) -> torch.Tensor:
    tokens = [ord(c) for c in text]
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(1)


def load_model(path: str) -> torch.nn.Module:
    model = torch.load(path, map_location="cpu")
    model.eval()
    return model


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Visualize attention weights for a text sequence"
    )
    parser.add_argument("--model", required=True, help="Path to a saved model")
    parser.add_argument("--input", required=True, help="Text file to analyze")
    parser.add_argument(
        "--out-dir", required=True, help="Directory for heatmaps"
    )
    args = parser.parse_args(argv)

    model = load_model(args.model)
    text = Path(args.input).read_text(encoding="utf-8")
    tokens = simple_tokenize(text)

    vis = AttentionVisualizer(model, out_dir=args.out_dir)
    vis.run(tokens)


if __name__ == "__main__":  # pragma: no cover - entry point
    main()
