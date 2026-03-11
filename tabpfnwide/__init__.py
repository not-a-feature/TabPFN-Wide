from .classifier import TabPFNWideClassifier
from tabpfn.architectures.base.attention.full_attention import MultiHeadAttention
from .patches import compute_attention_heads, _compute

# Apply patches to enable attention map extraction and fix compatibility
MultiHeadAttention.compute_attention_heads = compute_attention_heads
MultiHeadAttention._compute = _compute

__all__ = ["TabPFNWideClassifier"]
