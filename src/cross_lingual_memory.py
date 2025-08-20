import numpy as np
from typing import Iterable, Any, Tuple, List

try:  # optional torch dependency
    import torch
    from torch import nn
except Exception:  # pragma: no cover - allow running without torch
    from .torch_fallback import torch, nn

_cos_sim = nn.functional.cosine_similarity

from .hierarchical_memory import HierarchicalMemory
from .cross_lingual_translator import (
    CrossLingualTranslator,
    CrossLingualSpeechTranslator,
)
from .quantum_sampling import amplify_search
from .cross_lingual_utils import embed_text as _embed_text




class CrossLingualMemory(HierarchicalMemory):
    """HierarchicalMemory wrapper storing translations for text input."""

    def __init__(
        self,
        *args: Any,
        translator: CrossLingualTranslator | None = None,
        speech_translator: CrossLingualSpeechTranslator | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, translator=translator, **kwargs)
        self.speech_translator = speech_translator
        self.text_dim = self.compressor.encoder.in_features

    def add_texts(
        self, texts: Iterable[str], metadata: Iterable[Any] | None = None
    ) -> None:
        """Embed ``texts`` and add them plus translations."""
        base = list(texts)
        metas = list(metadata) if metadata is not None else base
        vecs: List[torch.Tensor] = []
        out_meta: List[Any] = []
        for t, m in zip(base, metas):
            v = _embed_text(t, self.text_dim)
            vecs.append(v)
            out_meta.append(m)
            if self.translator is not None:
                for trans in self.translator.translate_all(t).values():
                    vecs.append(_embed_text(trans, self.text_dim))
                    out_meta.append(trans)
        stacked = torch.stack(vecs)
        super().add(stacked, out_meta)

    def add_modalities(
        self,
        text: torch.Tensor | None = None,
        images: torch.Tensor | None = None,
        audio: torch.Tensor | None = None,
        sign: torch.Tensor | None = None,
        metadata: Iterable[Any] | None = None,
    ) -> None:  # type: ignore[override]
        """Add modality embeddings and store audio transcripts."""
        n = None
        for t in (text, images, audio, sign):
            if t is not None:
                n = t.shape[0]
                break
        if n is None:
            return
        if metadata is None:
            metas = [self._next_id + i for i in range(n)]
            self._next_id += n
        else:
            metas = list(metadata)
        if len(metas) != n:
            raise ValueError("metadata length mismatch")
        if text is not None:
            super().add(text, [{"id": m, "modality": "text"} for m in metas])
        if images is not None:
            super().add(images, [{"id": m, "modality": "image"} for m in metas])
        if audio is not None:
            super().add(audio, [{"id": m, "modality": "audio"} for m in metas])
            if self.speech_translator is not None:
                transcripts = [self.speech_translator.transcribe(a) for a in audio]
                self.add_texts(transcripts, metas)
        if sign is not None:
            super().add(sign, [{"id": m, "modality": "sign"} for m in metas])

    # ------------------------------------------------------------------
    # Convenience wrappers

    def add(
        self, x: torch.Tensor | str, metadata: Iterable[Any] | None = None
    ) -> None:  # type: ignore[override]
        """Add embeddings or raw text with translations."""
        if isinstance(x, str):
            self.add_texts([x], metadata)
        else:
            super().add(x, metadata)

    def search_text(self, text: str, k: int = 5, *, quantum: bool = False) -> Tuple[torch.Tensor, List[Any]]:
        """Search for ``text`` across languages."""
        queries = [text]
        langs = ["orig"]
        if self.translator is not None:
            for l, trans in self.translator.translate_all(text).items():
                queries.append(trans)
                langs.append(l)
        results: List[tuple[float, torch.Tensor, Any, str]] = []
        for q, lg in zip(queries, langs):
            q_vec = _embed_text(q, self.text_dim)
            vecs, meta = super().search(q_vec, k)
            scores = _cos_sim(vecs, q_vec.expand_as(vecs), dim=1)
            results.extend([(s.item(), v, m, lg) for s, v, m in zip(scores, vecs, meta)])
        if not results:
            return torch.empty(0, self.text_dim), []
        scores = [r[0] for r in results]
        tags = [r[3] for r in results]
        if quantum:
            order = amplify_search(scores, k, tags)
        else:
            order = np.argsort(scores)[::-1][:k]
        top = [results[i] for i in order]
        out_vecs = torch.stack([r[1] for r in top])
        out_meta = [r[2] for r in top]
        return out_vecs, out_meta

    def search_audio(
        self, audio: str | torch.Tensor, k: int = 5, *, quantum: bool = False
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Transcribe ``audio`` then search translations."""
        if self.speech_translator is None:
            return torch.empty(0, self.text_dim), []
        text = self.speech_translator.transcribe(audio)
        return self.search_text(text, k, quantum=quantum)

    def search(
        self, query: torch.Tensor | str, k: int = 5, *, quantum: bool = False
    ) -> Tuple[torch.Tensor, List[Any]]:  # type: ignore[override]
        """Search by embedding, text or audio path."""
        if isinstance(query, str):
            if query.endswith(".wav") and self.speech_translator is not None:
                return self.search_audio(query, k, quantum=quantum)
            return self.search_text(query, k, quantum=quantum)
        return super().search(query, k)


__all__ = ["CrossLingualMemory"]
