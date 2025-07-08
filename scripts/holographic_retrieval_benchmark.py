import argparse
import random
import torch
from asi.cross_modal_fusion import (
    CrossModalFusion,
    CrossModalFusionConfig,
    MultiModalDataset,
    retrieval_accuracy,
)
from asi.hierarchical_memory import HierarchicalMemory


def simple_tokenizer(text: str):
    return [ord(c) % 50 for c in text]


def generate_dataset(n: int, seq_len: int = 8):
    triples = []
    for _ in range(n):
        text = "".join(chr(97 + random.randint(0, 25)) for _ in range(seq_len))
        img = torch.randn(3, 16, 16)
        aud = torch.randn(1, 32)
        triples.append((text, img, aud))
    return triples


def main():
    parser = argparse.ArgumentParser(description="Holographic retrieval benchmark")
    parser.add_argument("--samples", type=int, default=1000, help="Number of triples")
    parser.add_argument("--latent-dim", type=int, default=64, help="Latent dimension")
    parser.add_argument("--k", type=int, default=1, help="Top-k retrieval")
    args = parser.parse_args()

    cfg = CrossModalFusionConfig(
        vocab_size=50,
        text_dim=args.latent_dim,
        img_channels=3,
        audio_channels=1,
        latent_dim=args.latent_dim,
    )
    model = CrossModalFusion(cfg)
    triples = generate_dataset(args.samples)
    dataset = MultiModalDataset(triples, simple_tokenizer)
    memory = HierarchicalMemory(
        dim=args.latent_dim,
        compressed_dim=args.latent_dim // 2,
        capacity=args.samples * 2,
        store_type="holographic",
    )

    acc = retrieval_accuracy(model, dataset, memory, batch_size=32, k=args.k)
    print(f"retrieval_accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
