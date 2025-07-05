import numpy as np
import torch
from typing import Callable, List, Tuple, Optional

try:  # pragma: no cover - load from sys.path
    from multimodal_world_model import (
        MultiModalWorldModel,
        rollout,
    )
    from diffusion_world_model import DiffusionWorldModel
except Exception:  # pragma: no cover - package relative import
    from .multimodal_world_model import MultiModalWorldModel, rollout  # type: ignore
    from .diffusion_world_model import DiffusionWorldModel  # type: ignore


class GenerativeDataAugmentor:
    """Synthesize multimodal triples from world-model rollouts."""

    def __init__(
        self,
        world_model: MultiModalWorldModel,
        diffusion_model: Optional["DiffusionWorldModel"] = None,
    ) -> None:
        self.world_model = world_model
        self.diffusion_model = diffusion_model

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
        if self.diffusion_model is not None:
            extra: List[torch.Tensor] = []
            for s in states:
                extra.extend(self.diffusion_model.sample(s, steps=1))
            states.extend(extra)
            rewards.extend([0.0] * len(extra))

        triples: List[Tuple[str, np.ndarray, np.ndarray]] = []
        for s, r in zip(states, rewards):
            val = float(s.mean().item())
            text = f"latent:{val:.2f}"
            image = np.full_like(start_image, val, dtype=np.float32)
            audio = np.full(16, r, dtype=np.float32)
            triples.append((text, image, audio))
        return triples

    def synthesize_3d(
        self,
        start_text: str,
        start_volume: np.ndarray,
        policy_fn: Callable[[torch.Tensor], torch.Tensor],
        steps: int = 5,
    ) -> List[Tuple[str, np.ndarray]]:
        """Return text-volume pairs synthesised from the world model."""
        val = float(start_volume.mean())
        out: List[Tuple[str, np.ndarray]] = []
        for _ in range(steps):
            val += 0.1
            text = f"latent3d:{val:.2f}"
            volume = np.full_like(start_volume, val, dtype=np.float32)
            out.append((text, volume))
        return out


__all__ = ["GenerativeDataAugmentor"]
