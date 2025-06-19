import torch


def flash_attention_3(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Placeholder for FlashAttention-3 (Plan.md S-2).

    This function falls back to PyTorch's scaled_dot_product_attention. Link the
    fused CUDA/ROCm kernel for production use.
    """
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)
