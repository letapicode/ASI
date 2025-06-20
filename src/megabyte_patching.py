import torch
from torch import nn

class MegaBytePatching(nn.Module):
    """Hierarchical byte patching for Plan.md C-4."""

    def __init__(self, patch_size: int, dim: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.byte_embed = nn.Embedding(256, dim)
        self.proj = nn.Linear(dim * patch_size, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed bytes in patches and project to the given dimension.

        Args:
            x: Integer tensor of shape (batch, seq) with values in [0, 255].
        Returns:
            Tensor of shape (batch, ceil(seq / patch_size), dim).
        """
        batch, seq = x.shape
        pad = (-seq) % self.patch_size
        if pad:
            x = torch.cat([x, torch.zeros(batch, pad, dtype=x.dtype, device=x.device)], dim=1)
        patches = x.unfold(1, self.patch_size, self.patch_size)  # (batch, num_patches, patch_size)
        emb = self.byte_embed(patches)
        emb = emb.reshape(batch, patches.size(1), -1)
        return self.proj(emb)
