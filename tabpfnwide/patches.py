from __future__ import annotations

from typing import Any

import torch

from tabpfn.architectures.base.attention.full_attention import MultiHeadAttention
from tabpfn.architectures.base.memory import support_save_peak_mem_factor
from tabpfn.architectures.encoders import (
    NormalizeFeatureGroupsEncoderStep,
    TorchPreprocessingPipeline,
)


class NormalizeAndPadFeatureGroupsEncoderStep(NormalizeFeatureGroupsEncoderStep):
    """Drop-in replacement for ``NormalizeFeatureGroupsEncoderStep`` that
    *also* zero-pads ``main`` (and optional auxiliary keys like
    ``nan_indicators``) up to ``num_features_per_group``.

    TabPFN 6.x performed both operations in a single
    ``VariableNumFeaturesEncoderStep`` (normalize_by_used_features → pad). The
    8.x pipeline only normalizes — the padding was dropped on the assumption
    that the runtime ``features_per_group`` always matches the encoder's
    ``num_features_per_group``. TabPFN-Wide breaks that assumption: the wide
    checkpoints were finetuned from the v2 base (encoder Linear sized for
    ``num_features_per_group=2``) but operate with ``features_per_group=1`` at
    the transformer level. This step restores the missing padding without
    changing module indices, so the wide state-dict still loads cleanly.

    The normalize step's ``num_features_per_group`` divisibility check is
    bypassed by overriding ``_fit`` to first pad ``main`` and ``aux_keys`` up
    to the target width, matching the 6.x behaviour.
    """

    aux_pad_keys: tuple[str, ...] = ("nan_indicators",)

    def _pad_to_target(self, x: torch.Tensor) -> torch.Tensor:
        missing = self.num_features_per_group - x.shape[-1]
        if missing <= 0:
            return x
        pad = torch.zeros(
            *x.shape[:-1],
            missing,
            device=x.device,
            dtype=x.dtype,
        )
        return torch.cat([x, pad], dim=-1)

    def _fit(self, state: dict[str, torch.Tensor], **kwargs: Any) -> None:
        # Pad the main key in the state dict so the parent's divisibility check
        # and ``number_of_used_features_`` computation see the full-width
        # tensor. The pad happens in-place on the state dict because the
        # following ``_transform`` pass re-reads it.
        main_key = self.in_keys[0]
        state[main_key] = self._pad_to_target(state[main_key])
        for key in self.aux_pad_keys:
            if key in state:
                state[key] = self._pad_to_target(state[key])
        super()._fit(state, **kwargs)

    def _transform(self, state: dict[str, torch.Tensor], **kwargs: Any) -> dict[str, torch.Tensor]:
        # _fit already padded the relevant keys in `state` when called above.
        # During pure-inference (no _fit), pad now.
        main_key = self.in_keys[0]
        if state[main_key].shape[-1] < self.num_features_per_group:
            state[main_key] = self._pad_to_target(state[main_key])
        for key in self.aux_pad_keys:
            if key in state and state[key].shape[-1] < self.num_features_per_group:
                state[key] = self._pad_to_target(state[key])
        return super()._transform(state, **kwargs)


def patch_encoder_for_narrow_feature_groups(
    model: torch.nn.Module,
    runtime_features_per_group: int,
) -> None:
    """Replace the encoder's ``NormalizeFeatureGroupsEncoderStep`` instances
    with ones that also zero-pad up to ``num_features_per_group``. No-op when
    there is no mismatch (i.e. the plain v2 model with features_per_group=2).

    Mutates ``model.encoder`` in place. Preserves module indices, so any
    state-dict already loaded into the model stays valid.
    """
    raw_encoder = model.encoder
    assert isinstance(raw_encoder, TorchPreprocessingPipeline), (
        f"Expected model.encoder to be a TorchPreprocessingPipeline, "
        f"got {type(raw_encoder).__name__}"
    )
    encoder: TorchPreprocessingPipeline = raw_encoder

    # Determine the encoder's intrinsic group width from the active normalize
    # step (i.e. ``normalize_by_used_features=True``). If that width already
    # matches the runtime, nothing to do.
    encoder_target = None
    for step in encoder:
        if isinstance(step, NormalizeFeatureGroupsEncoderStep) and step.normalize_by_used_features:
            encoder_target = step.num_features_per_group
            break

    if encoder_target is None or encoder_target <= runtime_features_per_group:
        return

    # Swap only the *active* NormalizeFeatureGroupsEncoderStep
    # (``normalize_by_used_features=True``) for the pad-aware subclass at the
    # same index. The legacy no-op step earlier in the pipeline is left alone
    # so it doesn't pad ``main`` before ``FeatureTransformEncoderStep`` — that
    # would feed an all-zero column into the per-feature mean/std fit and
    # poison the normalization.
    for idx, step in enumerate(encoder):
        if not isinstance(step, NormalizeFeatureGroupsEncoderStep):
            continue
        if not step.normalize_by_used_features:
            continue
        replacement = NormalizeAndPadFeatureGroupsEncoderStep(
            num_features_per_group=step.num_features_per_group,
            normalize_by_sqrt=step.normalize_by_sqrt,
            normalize_by_used_features=step.normalize_by_used_features,
            in_keys=step.in_keys,
            out_keys=step.out_keys,
        )
        encoder[idx] = replacement
        break


@support_save_peak_mem_factor  # type: ignore
def _compute(
    self,
    x: torch.Tensor,
    x_kv: torch.Tensor | None,
    k_cache: torch.Tensor | None,
    v_cache: torch.Tensor | None,
    kv_cache: torch.Tensor | None,
    *,
    cache_kv: bool,
    use_cached_kv: bool,
    reuse_first_head_kv: bool,
) -> torch.Tensor:
    """Attention computation that additionally records the attention map
    when ``self.save_att_map`` is set on the instance.

    Mirrors :meth:`MultiHeadAttention._compute` from upstream TabPFN, with an
    extra branch that materializes and averages the softmax-normalized
    attention map across heads.
    """
    q, k, v, kv, qkv = self.compute_qkv(
        x,
        x_kv,
        k_cache,
        v_cache,
        kv_cache,
        cache_kv=cache_kv,
        use_cached_kv=use_cached_kv,
        reuse_first_head_kv=reuse_first_head_kv,
    )

    if getattr(self, "save_att_map", False) and self.number_of_samples > 0:
        # Re-derive q/k from packed tensors, mirroring the static
        # ``compute_attention_heads`` so attention-map extraction works
        # regardless of whether QKV are packed.
        if qkv is not None:
            q_eff, k_eff, _v_eff = qkv.unbind(dim=-3)
        elif kv is not None:
            q_eff = q
            k_eff, _v_eff = kv.unbind(dim=-3)
        else:
            q_eff, k_eff, _v_eff = q, k, v

        assert q_eff is not None and k_eff is not None
        d_k = q_eff.shape[-1]
        attention_map_cpu = torch.zeros(q_eff.shape[1], k_eff.shape[1])
        for i in range(q_eff.shape[0]):
            q_slice = q_eff[i : i + 1]
            k_slice = k_eff[i : i + 1]

            logits = torch.einsum("b q h d, b k h d -> b q k h", q_slice, k_slice)
            logits *= (
                torch.sqrt(torch.tensor(1.0 / d_k)).to(k_eff.device)
                if self.softmax_scale is None
                else self.softmax_scale
            )
            attention_map = torch.softmax(logits, dim=2).detach().cpu()
            attention_map_cpu += attention_map.mean(-1).sum(0) / self.number_of_samples
            del attention_map, q_slice, k_slice

        if self.attention_map is None:
            self.attention_map = attention_map_cpu
        else:
            self.attention_map += attention_map_cpu

    attention_head_outputs = MultiHeadAttention.compute_attention_heads(
        q,
        k,
        v,
        kv,
        qkv,
        self.dropout_p,
        self.softmax_scale,
    )
    return torch.einsum(
        "... h d, h d s -> ... s",
        attention_head_outputs,
        self._w_out,
    )
