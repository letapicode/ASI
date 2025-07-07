"""Memory storing multi-modal summaries."""

from __future__ import annotations

from typing import Iterable, Any

import torch

from .hierarchical_memory import HierarchicalMemory
from .cross_modal_fusion import encode_all, MultiModalDataset, CrossModalFusion


class MultiModalSummaryMemory(HierarchicalMemory):
    """HierarchicalMemory with compressed image and audio summaries."""

    def __init__(
        self,
        *args: Any,
        image_summarizer: Any,
        audio_summarizer: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.image_summarizer = image_summarizer
        self.audio_summarizer = audio_summarizer

    def add_encoded(
        self,
        text: torch.Tensor,
        images: torch.Tensor,
        audio: torch.Tensor,
        metadata: Iterable[Any] | None = None,
    ) -> None:
        metas = (
            list(metadata)
            if metadata is not None
            else [self._next_id + i for i in range(text.size(0))]
        )
        self._next_id += text.size(0)
        for idx, (t, i, a, m) in enumerate(zip(text, images, audio, metas)):
            img_sum = self.image_summarizer.summarize(i.unsqueeze(0))
            aud_sum = self.audio_summarizer.summarize(a.unsqueeze(0))
            img_vec = self.image_summarizer.expand(img_sum)
            aud_vec = self.audio_summarizer.expand(aud_sum)
            fused = (t + img_vec + aud_vec) / 3.0
            super().add(
                fused.unsqueeze(0),
                [{"id": m, "image_summary": img_sum, "audio_summary": aud_sum}],
            )

    def add_dataset(
        self,
        model: CrossModalFusion,
        dataset: MultiModalDataset,
        batch_size: int = 8,
    ) -> None:
        text, images, audio, _ = encode_all(model, dataset, batch_size=batch_size)
        self.add_encoded(text, images, audio)


__all__ = ["MultiModalSummaryMemory"]
