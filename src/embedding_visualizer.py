from __future__ import annotations

import os
import tempfile
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Iterable, Optional, Sequence

import numpy as np

try:
    from umap import UMAP
    _HAS_UMAP = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_UMAP = False

from sklearn.manifold import TSNE
import plotly.express as px


class EmbeddingVisualizer:
    """Reduce embeddings and serve an interactive scatter plot."""

    def __init__(self, embeddings: np.ndarray, labels: Optional[Sequence[str]] = None) -> None:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2-D")
        self.embeddings = embeddings.astype(np.float32)
        self.labels = list(labels) if labels is not None else [str(i) for i in range(len(embeddings))]
        if len(self.labels) != len(self.embeddings):
            raise ValueError("labels length mismatch")
        self._reduced: Optional[np.ndarray] = None
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._html_file: Optional[str] = None

    def reduce(self, method: str = "umap", **kwargs) -> np.ndarray:
        """Reduce embeddings to 2D using UMAP or t-SNE."""
        if method == "umap" and _HAS_UMAP:
            reducer = UMAP(n_components=2, **kwargs)
            self._reduced = reducer.fit_transform(self.embeddings)
        else:
            reducer = TSNE(n_components=2, init="random", **kwargs)
            self._reduced = reducer.fit_transform(self.embeddings)
        return self._reduced

    def to_html(self, title: str = "Embeddings") -> str:
        if self._reduced is None:
            self.reduce("tsne")
        fig = px.scatter(x=self._reduced[:, 0], y=self._reduced[:, 1], text=self.labels)
        fig.update_layout(title=title)
        return fig.to_html(include_plotlyjs="cdn")

    def serve(self, port: int = 8000) -> None:
        """Serve the interactive plot on ``localhost``."""
        html = self.to_html()
        tmpdir = tempfile.mkdtemp(prefix="embvis_")
        self._html_file = os.path.join(tmpdir, "index.html")
        with open(self._html_file, "w", encoding="utf-8") as fh:
            fh.write(html)

        class Handler(SimpleHTTPRequestHandler):
            def log_message(self, format: str, *args: str) -> None:  # pragma: no cover - HTTP noise
                pass

        self._server = HTTPServer(("", port), Handler)

        def _run() -> None:
            os.chdir(tmpdir)
            self._server.serve_forever()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=1)
            self._thread = None
        if self._html_file and os.path.exists(os.path.dirname(self._html_file)):
            try:
                os.remove(self._html_file)
                os.rmdir(os.path.dirname(self._html_file))
            except OSError:
                pass


__all__ = ["EmbeddingVisualizer"]
