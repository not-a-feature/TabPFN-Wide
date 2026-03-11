import contextlib
from tabpfn.constants import XType, YType
from tabpfn.utils import infer_random_state
from tabpfn.base import determine_precision, create_inference_engine
from tabpfn.architectures.base.attention.full_attention import MultiHeadAttention
from tabpfn.architectures.base.memory import support_save_peak_mem_factor
from sklearn import config_context
from typing import Self
import torch

HAVE_FLASH_ATTN = False


@config_context(transform_output="default")  # type: ignore
def fit(self, X: XType, y: YType, model=None) -> Self:
    """Fit the model.

    Args:
        X: The input data.
        y: The target variable.
    """
    if not hasattr(self, "models_") or not self.differentiable_input:
        byte_size, rng = self._initialize_model_variables()
        ensemble_configs, X, y = self._initialize_dataset_preprocessing(X, y, rng)
    else:  # already fitted and prompt_tuning mode: no cat. features
        _, rng = infer_random_state(self.random_state)
        _, _, byte_size = determine_precision(self.inference_precision, self.devices_)

    if self.fit_mode == "batched":
        raise ValueError("The fit() function is currently not supported in the batched fit_mode.")

    # This line was added to allow a custom model to be passed to fit()
    if model:
        self.models_ = [model]

    # Initialize tuning attributes to defaults, as the original fit does
    self.tuned_classification_thresholds_ = None
    self.softmax_temperature_ = getattr(self, "softmax_temperature", 0.9)

    # Create the inference engine
    self.executor_ = create_inference_engine(
        X_train=X,
        y_train=y,
        models=self.models_,
        ensemble_configs=ensemble_configs,
        cat_ix=self.inferred_categorical_indices_,
        fit_mode=self.fit_mode,
        devices_=self.devices_,
        rng=rng,
        n_preprocessing_jobs=self.n_preprocessing_jobs,
        byte_size=byte_size,
        forced_inference_dtype_=self.forced_inference_dtype_,
        memory_saving_mode=self.memory_saving_mode,
        use_autocast_=self.use_autocast_,
        inference_mode=not self.differentiable_input,
    )

    return self


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
    use_second_set_of_queries: bool = False,
) -> torch.Tensor:
    """Attention computation.
    Called by 'forward', potentially on shards, once shapes have been normalized.
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
        # use_second_set_of_queries=use_second_set_of_queries,
    )
    attention_head_outputs = self.compute_attention_heads(
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


def compute_attention_heads(  # noqa: C901, PLR0912
    self,
    q: torch.Tensor | None,
    k: torch.Tensor | None,
    v: torch.Tensor | None,
    kv: torch.Tensor | None,
    qkv: torch.Tensor | None,
    dropout_p: float | None = None,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """This function was made a class method by adding 'self' as the first argument to allow access to class attributes"""
    assert (k is None) == (v is None)
    assert sum([qkv is None, kv is None, k is None and v is None]) == 2
    assert (qkv is None) != (q is None)

    if qkv is not None:
        q, k, v = qkv.unbind(dim=-3)
    elif kv is not None:
        k, v = kv.unbind(dim=-3)

    assert q is not None
    assert k is not None
    assert v is not None
    batch_size, seqlen_q, nhead, d_k = q.shape
    _, seqlen_kv, nhead_kv, d_v = v.shape
    share_kv_across_n_heads = nhead // nhead_kv
    if dropout_p is None:
        dropout_p = 0.0  # TODO: necessary?

    use_flash_attention = (
        HAVE_FLASH_ATTN
        and torch.cuda.is_available()
        and q.dtype == k.dtype == v.dtype == torch.float16
    )

    # this string comparison is reliable, as it does not compare to a subversion
    TORCH_2_ATTENTION_POSSIBLE = torch.__version__ >= "2" and torch.cuda.is_available()
    USE_TORCH_2_GQA = False
    if TORCH_2_ATTENTION_POSSIBLE:
        with contextlib.suppress(TypeError, RuntimeError):
            _ = torch.nn.functional.scaled_dot_product_attention(
                torch.empty(1, 1, 1, 1),
                torch.empty(1, 1, 1, 1),
                torch.empty(1, 1, 1, 1),
                enable_gqa=True,
            )
            USE_TORCH_2_GQA = True

    if use_flash_attention:

        def get_seqlen_cumsums(
            batch_size: int,
            seqlen: int,
            device: torch.device,
        ) -> torch.Tensor:
            return torch.arange(
                0,
                (batch_size + 1) * seqlen,
                step=seqlen,
                dtype=torch.int32,
                device=device,
            )

        if qkv is not None:
            attention_head_outputs = flash_attn_unpadded_qkvpacked_func(  # type: ignore
                qkv.reshape(batch_size * seqlen_q, 3, nhead, d_k),
                get_seqlen_cumsums(batch_size, seqlen_q, qkv.device),
                seqlen_q,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,  # defaults to 1/sqrt(d_k) if None
                causal=False,
                return_attn_probs=False,
                deterministic=False,
            )
        elif kv is not None:
            kv = MultiHeadAttention.broadcast_kv_across_heads(
                kv,
                share_kv_across_n_heads,
            )
            attention_head_outputs = flash_attn_unpadded_kvpacked_func(  # type: ignore
                q.reshape(batch_size * seqlen_q, nhead, d_k),
                kv.reshape(batch_size * seqlen_kv, 2, nhead, d_k),
                get_seqlen_cumsums(batch_size, seqlen_q, q.device),
                get_seqlen_cumsums(batch_size, seqlen_kv, kv.device),
                seqlen_q,
                seqlen_kv,
                dropout_p=dropout_p,
                causal=False,
                return_attn_probs=False,
                deterministic=False,
            )
        else:
            assert d_k <= d_v, (
                "This requirement is here for safety but not strictly necessary."
                "Needs testing/coding to remove."
            )
            if d_k < d_v:
                k = torch.nn.functional.pad(k, d_v - d_k)
                q = torch.nn.functional.pad(v, d_v - d_k)
                d_k_ = d_v
            k = MultiHeadAttention.broadcast_kv_across_heads(
                k,
                share_kv_across_n_heads,
            )
            v = MultiHeadAttention.broadcast_kv_across_heads(
                v,
                share_kv_across_n_heads,
            )
            attention_head_outputs = flash_attn_unpadded_func(  # type: ignore
                q.reshape(batch_size * seqlen_q, nhead, d_k_),  # type: ignore
                k.reshape(batch_size * seqlen_kv, nhead, d_k_),  # type: ignore
                v.reshape(batch_size * seqlen_kv, nhead, d_v),
                get_seqlen_cumsums(batch_size, seqlen_q, q.device),
                get_seqlen_cumsums(batch_size, seqlen_kv, k.device),
                seqlen_q,
                seqlen_kv,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=False,
                return_attn_probs=False,
                deterministic=False,
            )
    elif TORCH_2_ATTENTION_POSSIBLE:
        extra_inputs = {}
        if softmax_scale is not None:
            extra_inputs["scale"] = softmax_scale  # defaults to 1/sqrt(d_k) if None or not provided
        if not USE_TORCH_2_GQA:
            k = MultiHeadAttention.broadcast_kv_across_heads(
                k,
                share_kv_across_n_heads,
            )
            v = MultiHeadAttention.broadcast_kv_across_heads(
                v,
                share_kv_across_n_heads,
            )
        else:
            extra_inputs["enable_gqa"] = True

        ### This part was added to extract attention maps
        if getattr(self, "save_att_map", False) and self.number_of_samples > 0:
            attention_map_cpu = torch.zeros(q.shape[1], k.shape[1])
            for i in range(0, v.shape[0], 1):
                q_slice = q[i : i + 1]
                k_slice = k[i : i + 1]

                logits = torch.einsum("b q h d, b k h d -> b q k h", q_slice, k_slice)
                logits *= (
                    torch.sqrt(torch.tensor(1.0 / d_k)).to(k.device)
                    if softmax_scale is None
                    else softmax_scale
                )
                attention_map = torch.softmax(logits, dim=2).detach().cpu()

                attention_map_cpu += attention_map.mean(-1).sum(0) / self.number_of_samples
                del attention_map
                del q_slice
                del k_slice
                torch.cuda.empty_cache()
            if self.attention_map is None:
                self.attention_map = attention_map_cpu
            else:
                self.attention_map += attention_map_cpu
        ### End of added part

        attention_head_outputs = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            dropout_p=dropout_p,
            **extra_inputs,
        )
        attention_head_outputs = attention_head_outputs.transpose(1, 2)
    else:
        ### This part was added to extract attention maps
        if getattr(self, "save_att_map", False) and self.number_of_samples > 0:
            attention_map_cpu = torch.zeros(q.shape[1], k.shape[1])
            for i in range(0, v.shape[0]):
                q_slice = q[i : i + 1]
                k_slice = k[i : i + 1]

                logits = torch.einsum("b q h d, b k h d -> b q k h", q_slice, k_slice)
                logits *= (
                    torch.sqrt(torch.tensor(1.0 / d_k)).to(k.device)
                    if softmax_scale is None
                    else softmax_scale
                )
                attention_map = torch.softmax(logits, dim=2).detach().cpu()

                attention_map_cpu += attention_map.mean(-1).sum(0) / self.number_of_samples
                del attention_map
                del q_slice
                del k_slice
                torch.cuda.empty_cache()
            if self.attention_map is None:
                self.attention_map = attention_map_cpu
            else:
                self.attention_map += attention_map_cpu
        ### End of added part

        k = MultiHeadAttention.broadcast_kv_across_heads(k, share_kv_across_n_heads)
        v = MultiHeadAttention.broadcast_kv_across_heads(v, share_kv_across_n_heads)
        logits = torch.einsum("b q h d, b k h d -> b q k h", q, k)
        logits *= (
            torch.sqrt(torch.tensor(1.0 / d_k)).to(k.device)
            if softmax_scale is None
            else softmax_scale
        )
        ps = torch.softmax(logits, dim=2)
        ps = torch.dropout(ps, dropout_p, train=True)
        attention_head_outputs = torch.einsum("b q k h, b k h d -> b q h d", ps, v)

    return attention_head_outputs.reshape(
        batch_size,
        seqlen_q,
        nhead,
        d_v,
    )
