from .classifier import TabPFNWideClassifier
from tabpfn.architectures.base.attention.full_attention import MultiHeadAttention

from .patches import _compute

# Apply patches to enable attention map extraction and fix compatibility
MultiHeadAttention._compute = _compute

__all__ = ["TabPFNWideClassifier"]
