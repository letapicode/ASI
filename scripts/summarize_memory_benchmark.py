import argparse
import json
import torch

from asi.summarizing_memory import SummarizingMemory


def main(n: int = 100):
    mem = SummarizingMemory(dim=4, compressed_dim=2, capacity=n, summary_threshold=2)
    data = torch.randn(n, 4)
    mem.add(data, metadata=[f"v{i}" for i in range(n)])
    before = len(mem.compressor.buffer.data)
    mem.summarize(lambda x: "sum")
    after = len(mem.compressor.buffer.data)
    print(json.dumps({"before": before, "after": after}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarizing memory benchmark")
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()
    main(args.n)
