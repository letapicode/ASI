import argparse
import urllib.request
from pathlib import Path

import torch
from torch import nn

from asi.rwkv_loop import RWKVLoop
from asi.hierarchical_memory import HierarchicalMemory
from asi.distributed_memory import DistributedMemory
from asi.memory_service import serve
from asi.link_slot_attention import LinkSlotAttention
from asi.chunkwise_retrainer import ChunkWiseRetrainer


def load_dataset(path: str = "data/tinyshakespeare.txt") -> tuple[torch.Tensor, int]:
    """Download and tokenize TinyShakespeare if ``path`` is missing."""
    url = (
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
        "tinyshakespeare/input.txt"
    )
    p = Path(path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, p)
    text = p.read_text(encoding="utf-8")
    vocab = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    tokens = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return tokens, len(vocab)


def evaluate(model: nn.Module, seq: torch.Tensor, chunk: int = 64) -> tuple[float, float, float, int]:
    """Return loss, perplexity, hit rate and memory size."""
    criterion = nn.CrossEntropyLoss()
    hits = 0
    attempts = 0
    total = 0.0
    count = 0

    orig = model.memory.search

    def search(query: torch.Tensor, k: int = 5):
        nonlocal hits, attempts
        out, meta = orig(query, k)
        n = query.size(0) if query.ndim == 2 else 1
        attempts += n
        if out.numel() > 0:
            hits += n
        return out, meta

    model.memory.search = search  # type: ignore

    with torch.no_grad():
        for i in range(0, seq.size(0) - chunk, chunk):
            inp = seq[i : i + chunk].unsqueeze(0)
            tgt = seq[i + 1 : i + 1 + chunk].unsqueeze(0)
            logits = model(inp)
            loss = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1))
            total += loss.item()
            count += 1

    model.memory.search = orig  # type: ignore
    avg_loss = total / max(count, 1)
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    hit_rate = hits / max(attempts, 1)
    mem_size = len(model.memory.store)
    return avg_loss, ppl, hit_rate, mem_size


def save_checkpoint(model: nn.Module, optim: torch.optim.Optimizer, path: str, step: int) -> None:
    """Persist model parameters and hierarchical memory."""
    ckpt = Path(path) / f"step{step}"
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optim": optim.state_dict()}, ckpt / "model.pt")
    model.memory.save(ckpt / "memory")


class InfiniteContextModel(nn.Module):
    """Toy language model combining recurrence and retrieval."""

    def __init__(
        self, vocab_size: int = 100, dim: int = 16, remotes: list[str] | None = None
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.loop = RWKVLoop(dim)
        if remotes:
            self.memory = DistributedMemory(
                dim=dim, compressed_dim=dim // 2, capacity=50, remotes=remotes
            )
        else:
            self.memory = HierarchicalMemory(dim=dim, compressed_dim=dim // 2, capacity=50)
        self.link = LinkSlotAttention(self.memory, dim=dim, k_top=2)
        self.out = nn.Linear(dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.loop(x)
        x = self.link(x)
        return self.out(x)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train toy infinite-context model")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to store checkpoints",
    )
    parser.add_argument(
        "--remote-memory",
        nargs="*",
        help="Addresses of remote memory servers",
    )
    parser.add_argument(
        "--serve-memory",
        type=str,
        help="Expose local memory over gRPC at this address",
    )
    args = parser.parse_args()

    seq, vocab = load_dataset()

    torch.manual_seed(0)
    model = InfiniteContextModel(vocab_size=vocab, dim=32, remotes=args.remote_memory)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = ChunkWiseRetrainer(model, optimizer, chunk_size=64)

    server = None
    if args.serve_memory:
        server = serve(model.memory, args.serve_memory)

    for epoch in range(1, args.epochs + 1):
        loss = trainer.train([seq], epochs=1)
        print(f"epoch {epoch} loss {loss:.4f}")
        e_loss, ppl, hit, mem = evaluate(model, seq)
        print(
            f"eval ppl {ppl:.2f} | memory {mem} | hit rate {hit:.2f}"
        )
        save_checkpoint(model, optimizer, args.checkpoint_dir, epoch)

    if server is not None:
        server.stop(0)


if __name__ == "__main__":
    main()
