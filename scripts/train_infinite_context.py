import argparse
import torch
from torch import nn

from asi.rwkv_loop import RWKVLoop
from asi.hierarchical_memory import HierarchicalMemory
from asi.link_slot_attention import LinkSlotAttention
from asi.chunkwise_retrainer import ChunkWiseRetrainer


class InfiniteContextModel(nn.Module):
    """Toy language model combining recurrence and retrieval."""

    def __init__(self, vocab_size: int = 100, dim: int = 16, use_async: bool = False) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.loop = RWKVLoop(dim)
        self.memory = HierarchicalMemory(dim=dim, compressed_dim=dim // 2, capacity=50, use_async=use_async)
        self.link = LinkSlotAttention(self.memory, dim=dim, k_top=2)
        self.out = nn.Linear(dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.loop(x)
        x = self.link(x)
        return self.out(x)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train toy infinite-context model")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs per chunk")
    parser.add_argument("--async-memory", action="store_true", help="Use async hierarchical memory")
    args = parser.parse_args()

    torch.manual_seed(0)
    model = InfiniteContextModel(use_async=args.async_memory)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = ChunkWiseRetrainer(model, optimizer, chunk_size=8)

    # Dummy token stream
    seq = torch.randint(0, 100, (64,), dtype=torch.long)

    loss = trainer.train([seq], epochs=args.epochs)
    print(f"avg loss: {loss:.4f}")


if __name__ == "__main__":
    main()
