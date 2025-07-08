import argparse
from pathlib import Path
import numpy as np
from asi.holographic_vector_store import HolographicVectorStore


def migrate(input_path: str, output_path: str) -> None:
    input_path = Path(input_path)
    data = np.load(input_path, allow_pickle=True)
    dim = int(data["vectors"].shape[1])
    store = HolographicVectorStore(dim)
    for vec, meta in zip(data["vectors"], data["meta"].tolist()):
        enc = store.encode(vec, np.zeros(dim, dtype=np.float32), np.zeros(dim, dtype=np.float32))
        store.add(enc[None], metadata=[meta])
    store.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert vector dumps to HRR format")
    parser.add_argument("input", help="Path to existing store .npz file")
    parser.add_argument("output", help="Output directory for HRR store")
    args = parser.parse_args()
    migrate(args.input, args.output)


if __name__ == "__main__":
    main()
