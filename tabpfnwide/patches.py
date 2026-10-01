from __future__ import annotations

import math

import torch
from torch import nn

from tabpfn.architectures.tabpfn_v2 import AlongRowAttention, TabPFNV2


def narrow_feature_group_embedder(model: TabPFNV2) -> None:
    """Switch a ``features_per_group=2`` v2 model to one feature per group.

    The wide checkpoints were finetuned from the v2 base, whose input projection
    ``feature_group_embedder`` (``Linear(4, emsize)``) consumes
    ``[x_1, x_2, nan_1, nan_2]``, but they operate with ``features_per_group=1``.
    TabPFN <=8 handled this by zero-padding each group to ``[x, 0]`` and ``[nan, 0]``
    before the projection; the feature-group normalization then scaled ``x`` by
    ``sqrt(2 / 1)``. TabPFN 9 derives the projection width from
    ``features_per_group`` and no longer pads, so this keeps the projection columns
    for ``x_1`` and ``nan_1`` and folds the ``sqrt(2)`` scale into the ``x_1``
    column, which reproduces the padded computation exactly.

    Mutates ``model`` in place. Call after the checkpoint has been loaded, since
    the narrowed projection no longer matches the checkpoint's weight shape.
    """
    weight = model.feature_group_embedder.weight
    assert weight.shape[1] == 4, (
        f"Expected a features_per_group=2 projection with 4 input columns, "
        f"got weight of shape {tuple(weight.shape)}"
    )
    narrow = nn.Linear(2, weight.shape[0], bias=False, device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        narrow.weight.copy_(torch.stack([weight[:, 0] * math.sqrt(2), weight[:, 2]], dim=1))
    model.feature_group_embedder = narrow
    model.features_per_group = 1


def forward_recording_attention(self: AlongRowAttention, x_BrSE: torch.Tensor) -> torch.Tensor:
    """Between-features attention that additionally records the attention map.

    Accumulates the softmax-normalized attention map, averaged across heads and
    summed over rows divided by ``self.number_of_samples``, into
    ``self.attention_map``. Rows are processed one at a time on the input's
    device because the full ``(rows, heads, features, features)`` tensor does not
    fit in memory for wide data; only the accumulated ``(features, features)`` map
    is moved to the CPU. The output is computed by the upstream
    :meth:`AlongRowAttention.forward`.
    """
    Br, C, _ = x_BrSE.shape
    with torch.no_grad():
        q_BrHCD = self.q_projection(x_BrSE).view(Br, C, -1, self.head_dim).transpose(1, 2)
        k_BrHCD = self.k_projection(x_BrSE).view(Br, C, -1, self.head_dim).transpose(1, 2)
        attention_sum_CC = torch.zeros(C, C, device=x_BrSE.device, dtype=torch.float32)
        for i in range(Br):
            logits_HCC = q_BrHCD[i] @ k_BrHCD[i].transpose(-1, -2) / math.sqrt(self.head_dim)
            attention_sum_CC += torch.softmax(logits_HCC.float(), dim=-1).mean(0)
        attention_map_cpu = (attention_sum_CC / self.number_of_samples).cpu()
        del q_BrHCD, k_BrHCD, attention_sum_CC

    if self.attention_map is None:
        self.attention_map = attention_map_cpu
    else:
        self.attention_map += attention_map_cpu

    return AlongRowAttention.forward(self, x_BrSE)
