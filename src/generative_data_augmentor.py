import numpy as np
import torch
from typing import Callable, List, Tuple

try:  # pragma: no cover - load from sys.path
    from multimodal_world_model import MultiModalWorldModel, rollout
except Exception:  # pragma: no cover - package relative import
    from .multimodal_world_model import MultiModalWorldModel, rollout  # type: ignore


class GenerativeDataAugmentor:
    """Synthesize multimodal triples from world-model rollouts."""

    def __init__(self, world_model: MultiModalWorldModel) -> None:
        self.world_model = world_model

    def _tokenize(self, text: str) -> torch.Tensor:
        tokens = [ord(c) % self.world_model.cfg.vocab_size for c in text]
        return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

    def synthesize(
        self,
        start_text: str,
        start_image: np.ndarray,
        policy_fn: Callable[[torch.Tensor], torch.Tensor],
        steps: int = 5,
    ) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """Roll out ``world_model`` and return text-image-audio triples."""
        t = self._tokenize(start_text)
        img = torch.tensor(start_image, dtype=torch.float32).unsqueeze(0)
        states, rewards = rollout(self.world_model, t, img, policy_fn, steps=steps)
        triples: List[Tuple[str, np.ndarray, np.ndarray]] = []
        for s, r in zip(states, rewards):
            val = float(s.mean().item())
            text = f"latent:{val:.2f}"
            image = np.full_like(start_image, val, dtype=np.float32)
            audio = np.full(16, r, dtype=np.float32)
            triples.append((text, image, audio))
        return triples


__all__ = ["GenerativeDataAugmentor"]
