from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def cluster_texts(texts: Iterable[str], n_clusters: int = 5, top_k: int = 5) -> List[str]:
    """Cluster ``texts`` and return short summaries per cluster."""
    texts = [t for t in texts if t.strip()]
    if not texts:
        return []
    vec = TfidfVectorizer(max_features=4096, stop_words="english")
    mat = vec.fit_transform(texts)
    k = min(n_clusters, mat.shape[0])
    km = KMeans(n_clusters=k, n_init="auto", random_state=0)
    labels = km.fit_predict(mat)
    terms = np.array(vec.get_feature_names_out())
    summaries: List[str] = []
    for i in range(k):
        mask = labels == i
        if not np.any(mask):
            summaries.append("")
            continue
        centroid = mat[mask].mean(axis=0)
        idx = np.argsort(centroid.toarray().ravel())[::-1][:top_k]
        summaries.append(" ".join(terms[idx]))
    return summaries


def summarize_dataset(root: str | Path, ext: str = ".txt", n_clusters: int = 5) -> List[str]:
    """Return content summaries for text files under ``root``."""
    paths = sorted(Path(root).rglob(f"*{ext}"))
    texts = [Path(p).read_text(errors="ignore") for p in paths]
    return cluster_texts(texts, n_clusters=n_clusters)


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize dataset content")
    parser.add_argument("path", help="Dataset root directory")
    parser.add_argument("--clusters", type=int, default=5)
    parser.add_argument("--ext", default=".txt")
    args = parser.parse_args(argv)
    summaries = summarize_dataset(args.path, args.ext, args.clusters)
    for s in summaries:
        print(f"- {s}")


if __name__ == "__main__":  # pragma: no cover - CLI helper
    main()
