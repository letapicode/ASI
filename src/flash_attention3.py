import torch

try:
    from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func
    _HAS_FLASH3 = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_FLASH3 = False

# Expose availability for other modules to use the fused kernel


def flash_attention_3(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Call FlashAttention-3 if available, else fall back to PyTorch.

    The wrapper preserves the exact soft-max semantics and works on CUDA/ROCm
    builds. When the kernel cannot be imported, it defaults to
    ``torch.nn.functional.scaled_dot_product_attention``. Availability of the
    fused kernel is signaled by the ``_HAS_FLASH3`` flag.
    """
    if _HAS_FLASH3:
        qkv = torch.stack([q, k, v], dim=2)
        return flash_attn_qkvpacked_func(qkv, causal=False)
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)
