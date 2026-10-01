from __future__ import annotations

import os
import types

import numpy as np
import torch
from scipy.sparse import csr_matrix, diags

from tabpfn import TabPFNClassifier
from tabpfn.base import ClassifierModelSpecs
from tabpfn.model_loading import load_model_criterion_config

from tabpfnwide.patches import (
    forward_recording_attention,
    forward_recording_label_attention,
    narrow_feature_group_embedder,
)

VALID_MODELS = [
    "v2",
    "wide-v2-1.5k",
    "wide-v2-1.5k-nocat",
    "wide-v2-5k",
    "wide-v2-5k-nocat",
    "wide-v2-8k",
    "wide-v2-8k-nocat",
]


class TabPFNWideClassifier(TabPFNClassifier):
    def __init__(
        self,
        model_name="",
        model_path="",
        device="cuda",
        n_estimators=1,
        features_per_group=1,
        save_attention_maps=False,
        **kwargs,
    ):

        # Check arguments
        if (model_name and model_path) or (not model_name and not model_path):
            raise ValueError("Either model_name or model_path must be specified, but not both.")

        if model_name:
            if model_name not in VALID_MODELS:
                raise ValueError(
                    f"Model name {model_name} not recognized. Choose from {VALID_MODELS}"
                )
            if model_name != "v2":
                model_path = self._get_model_path(model_name)

        if model_name != "v2" and not os.path.isfile(model_path):
            raise ValueError(f"Model path {model_path} does not exist.")

        # save_attention_maps: False, True (full feature-by-feature maps), or
        # "label" (only the label's attention to each feature, per row).
        if save_attention_maps not in (False, True, "label"):
            raise ValueError(
                f"save_attention_maps must be False, True or 'label', got {save_attention_maps!r}"
            )
        if save_attention_maps and (n_estimators != 1 or features_per_group != 1):
            raise ValueError(
                "save_attention_maps can only be set when n_estimators=1 and features_per_group=1"
            )

        # Build a ClassifierModelSpecs that bundles the (potentially custom)
        # wide checkpoint together with the v2 architecture/inference config.
        model_specs = self._build_model_specs(
            model_name=model_name,
            model_path=model_path,
            features_per_group=features_per_group,
            device=device,
        )

        if "ignore_pretraining_limits" not in kwargs:
            kwargs["ignore_pretraining_limits"] = True

        super().__init__(
            device=device,
            n_estimators=n_estimators,
            model_path=model_specs,
            **kwargs,
        )

        # Keep wide-specific attributes for downstream code and for diagnostics.
        self.model_name = model_name
        self.wide_model_path = model_path
        self.features_per_group = features_per_group
        self.save_attention_maps = save_attention_maps
        self._wide_model = model_specs.model

        # Install the attention-recording patch once, per attention instance,
        # only on this model. The buffer reset and number_of_samples update
        # happen later in fit()/_predict_proba().
        if self.save_attention_maps:
            recorder = (
                forward_recording_label_attention
                if self.save_attention_maps == "label"
                else forward_recording_attention
            )
            for block in self._wide_model.blocks:
                attn = block.per_sample_attention_between_features
                attn.forward = types.MethodType(recorder, attn)

    @staticmethod
    def _build_model_specs(model_name, model_path, features_per_group, device):
        """Load the v2 architecture, swap in the wide checkpoint, and wrap the
        result in a ClassifierModelSpecs for injection into TabPFNClassifier.
        """
        # Validate the wide checkpoint path *before* paying for the v2 download
        # so configuration mistakes surface immediately.
        if model_name != "v2" and (not model_path or not os.path.isfile(model_path)):
            raise ValueError(
                f"Wide checkpoint path must point to an existing file when "
                f"model_name != 'v2', got model_name={model_name!r}, model_path={model_path!r}."
            )

        models, _, configs, inference_config = load_model_criterion_config(
            model_path=None,
            check_bar_distribution_criterion=False,
            cache_trainset_representation=False,
            estimator_type="classifier",
            version="v2",
            download_if_not_exists=True,
        )
        model = models[0]
        config = configs[0]

        # Load the wide checkpoint *before* narrowing the input projection, so
        # that its weight still matches the checkpoint's (emsize, 4) shape.
        if model_name != "v2":
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)

        # @Master Students: The wide checkpoints were finetuned from the v2 base
        # (input projection sized for ``features_per_group=2``) but operate with
        # ``features_per_group=1``. TabPFN 9 sizes the projection from
        # ``features_per_group`` and no longer zero-pads groups, so the
        # projection is narrowed to the equivalent one-feature form.
        if features_per_group == 1:
            narrow_feature_group_embedder(model)
        config.features_per_group = features_per_group

        return ClassifierModelSpecs(
            model=model,
            architecture_config=config,
            inference_config=inference_config,
        )

    def _get_model_path(self, model_name):
        """Get the local path for a model, downloading it if necessary."""
        # Use a standard cache directory
        cache_dir = os.path.join(os.path.expanduser("~"), ".tabpfnwide", "models")
        os.makedirs(cache_dir, exist_ok=True)

        filename = f"tabpfn-{model_name}.pt"
        local_path = os.path.join(cache_dir, filename)

        if not os.path.exists(local_path):
            # Define the URL dynamically based on package version
            try:
                from importlib.metadata import version

                package_version = version("tabpfnwide")
            except Exception:
                package_version = "0.1.0"

            # Ensure the tag starts with v
            tag_version = (
                package_version if package_version.startswith("v") else f"v{package_version}"
            )

            url = f"https://github.com/not-a-feature/TabPFN-Wide/releases/download/{tag_version}/{filename}"

            print(f"Downloading model {model_name} from {url} to {local_path}...")
            try:
                torch.hub.download_url_to_file(url, local_path, progress=True)
            except Exception as e:
                # Clean up partial download if it failed
                if os.path.exists(local_path):
                    os.remove(local_path)
                raise RuntimeError(
                    f"Failed to download model {model_name} from {url}. "
                    f"Please check your internet connection or manually download the model "
                    f"to {local_path}"
                ) from e

        return local_path

    def fit(self, X, y):
        # Store n_features_in_ for attention map cropping.
        self.n_features_in_ = X.shape[1]

        if self.save_attention_maps:
            for block in self._wide_model.blocks:
                block.per_sample_attention_between_features.number_of_samples = X.shape[0]

        return super().fit(X, y)

    def _predict_proba(self, X):
        # Reset attention buffers before each forward pass so they reflect only
        # this call. Without this, repeated predict/predict_proba invocations
        # would accumulate into the same buffer.
        if self.save_attention_maps:
            for block in self._wide_model.blocks:
                block.per_sample_attention_between_features.attention_map = None
                block.per_sample_attention_between_features.label_attention = []

        return super()._predict_proba(X)

    @property
    def model(self):
        """The wide model used by this estimator.

        Returns the model bound to the parent ``models_`` list once ``fit()``
        has been called, falling back to the pre-loaded wide model otherwise.
        """
        if hasattr(self, "models_") and self.models_:
            return self.models_[0]
        return self._wide_model

    def get_attention_maps(self):
        """Return attention maps mapped to original input features.

        Each token is assigned to the input feature it was derived from, following
        the fitted preprocessing pipeline (see :meth:`_get_feature_mapping_info`).
        Attention between two input features is the mean over all pairs of their
        tokens. Tokens added by the pipeline (SVD, fingerprint) are left out.

        Returns:
            List of numpy arrays, one per transformer layer, with shape
            (n_features_in_, n_features_in_) representing attention between
            original input features. Rows and columns of features the
            pipeline dropped (e.g. constant columns) are NaN.
        """
        if self.save_attention_maps is not True:
            raise ValueError(
                f"Full attention maps require save_attention_maps=True, "
                f"got {self.save_attention_maps!r}."
            )

        raw_maps = []
        for block in self.model.blocks:
            attn = getattr(block.per_sample_attention_between_features, "attention_map", None)
            if attn is not None:
                raw_maps.append(attn.numpy())

        mapping_info = self._get_feature_mapping_info()
        n_tokens = mapping_info["n_preprocessed"]
        token_features = mapping_info["token_features"]

        # P[i, p] = 1 if token p was derived from input feature i, row-normalized.
        cols = [p for p, feature in enumerate(token_features) if feature is not None]
        rows = [token_features[p] for p in cols]
        P = csr_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(self.n_features_in_, n_tokens)
        )
        tokens_per_feature = np.asarray(P.sum(axis=1)).ravel()
        dropped = tokens_per_feature == 0
        P_norm = diags(1.0 / np.where(dropped, 1.0, tokens_per_feature)) @ P

        mapped_maps = []
        for raw_attn in raw_maps:
            # The last token is the label.
            assert raw_attn.shape == (n_tokens + 1, n_tokens + 1), (
                f"Attention map of shape {raw_attn.shape} does not match "
                f"{n_tokens} feature tokens plus the label token."
            )
            mapped_attn = np.asarray((P_norm @ raw_attn[:n_tokens, :n_tokens]) @ P_norm.T)
            mapped_attn[dropped, :] = np.nan
            mapped_attn[:, dropped] = np.nan
            mapped_maps.append(mapped_attn)

        return mapped_maps

    def _get_feature_mapping_info(self):
        """Map each input feature to the token positions it occupies in the model input.

        Follows the final feature schema of the fitted preprocessing pipeline, which
        records every reordering, removal and addition of columns: the shuffle, the
        move of categorical columns to the front, dropped constant columns, the
        transformed copies appended by ``append_to_original`` and the SVD and
        fingerprint features. A token belongs to an input feature if its column
        carries that feature's name, either as its own name (passed through or
        encoded one-to-one) or as its ``ancestor`` (distribution-transformed).
        Tokens added by the pipeline belong to no feature. Any other token, such as
        an expanded one-hot column, raises an error.

        Returns a dict with:
        - original_to_preprocessed: input feature index -> list of token positions,
          empty for features the pipeline dropped
        - n_preprocessed: number of feature tokens (the label token excluded)
        - token_features: input feature index per token position, None for tokens
          added by the pipeline
        - token_names: column name per token position
        """
        if self.inference_config_.ENABLE_GPU_PREPROCESSING:
            raise ValueError(
                "Attention cannot be mapped to input features with GPU preprocessing, "
                "which reorders and adds columns outside the recorded feature schema."
            )
        (member,) = self.executor_.ensemble_members
        pipeline = member.cpu_preprocessor

        input_index = {f.name: j for j, f in enumerate(self.inferred_feature_schema_.features)}
        assert len(input_index) == self.n_features_in_, (
            f"Expected {self.n_features_in_} uniquely named input features, "
            f"got {len(input_index)}."
        )
        added_prefixes = tuple(step.added_feature_prefix() for step, _ in pipeline.steps)

        token_names, token_features = [], []
        for column in pipeline.final_feature_schema_.features:
            source = column.name if column.name in input_index else column.ancestor
            if source in input_index:
                token_features.append(input_index[source])
            elif (column.ancestor or column.name).startswith(added_prefixes):
                token_features.append(None)
            else:
                raise ValueError(
                    f"Cannot map token column {column.name!r} (ancestor {column.ancestor!r}) "
                    f"to an input feature."
                )
            token_names.append(column.name)

        original_to_preprocessed = {j: [] for j in range(self.n_features_in_)}
        for position, feature in enumerate(token_features):
            if feature is not None:
                original_to_preprocessed[feature].append(position)

        return {
            "original_to_preprocessed": original_to_preprocessed,
            "n_preprocessed": len(token_features),
            "token_features": token_features,
            "token_names": token_names,
        }

    def get_raw_attention_maps(self):
        """Return raw attention maps without any processing.

        Returns a tuple of (maps, n_features_in) where maps is a list of raw
        attention matrices and n_features_in is the number of input features.
        """
        if self.save_attention_maps is not True:
            raise ValueError(
                f"Full attention maps require save_attention_maps=True, "
                f"got {self.save_attention_maps!r}."
            )

        maps = []
        for block in self.model.blocks:
            attn = getattr(block.per_sample_attention_between_features, "attention_map", None)
            if attn is not None:
                maps.append(attn.numpy())

        n_features = getattr(self, "n_features_in_", None)
        return maps, n_features

    def get_feature_mapping_debug(self):
        """Get detailed feature mapping info for debugging.

        Returns the mapping info dict showing how original features
        map to preprocessed token positions.
        """
        return self._get_feature_mapping_info()

    def get_attention_to_label(self, aggregation=np.mean):
        """Return attention scores showing how much the label attends to each feature.

        In TabPFN's architecture, the label (y) is concatenated as the last token
        in the feature dimension. This method extracts how much the label token
        (as a query) attends to each feature (as keys), indicating feature
        importance for the prediction.

        Args:
            aggregation: Function combining the per-row scores over all rows
                (train and test) of the last predict call, called as
                ``aggregation(scores, axis=0)`` on a ``(rows, tokens)`` array,
                e.g. ``np.mean``, ``np.median`` or
                ``functools.partial(np.quantile, q=0.9)``. Anything other than
                ``np.mean`` requires ``save_attention_maps="label"``. With
                ``save_attention_maps=True`` the scores come from the full maps,
                which hold the sum over rows divided by the number of training
                rows.

        Returns:
            numpy array of shape (n_features_in_,) where each value represents
            how much the label token attends to that original input feature,
            averaged across all transformer layers and over the feature's
            tokens. NaN for features the preprocessing dropped (e.g. constant
            columns), which have no token.
        """
        if not self.save_attention_maps:
            raise ValueError("Attention maps are not being recorded (save_attention_maps=False).")
        if aggregation is not np.mean and self.save_attention_maps != "label":
            raise ValueError(
                "Aggregations other than np.mean require save_attention_maps='label', "
                "since the full maps only keep the summed rows."
            )

        mapping_info = self._get_feature_mapping_info()

        # Collect attention from label to features from each layer
        layer_attentions = []
        for block in self.model.blocks:
            attn_module = block.per_sample_attention_between_features
            if self.save_attention_maps == "label":
                assert attn_module.label_attention, "No label attention recorded; predict first."
                rows_by_tokens = torch.cat(attn_module.label_attention).numpy()
                per_token = np.asarray(aggregation(rows_by_tokens, axis=0))
                assert per_token.shape == rows_by_tokens.shape[1:], (
                    f"aggregation must reduce (rows, tokens) to (tokens,), got {per_token.shape}"
                )
                # Drop the last token (label-to-label self-attention).
                layer_attentions.append(per_token[:-1])
                continue
            attn = getattr(attn_module, "attention_map", None)
            if attn is not None:
                raw_attn = attn.numpy()
                # Last ROW = how much the label (query) attends to each feature (key)
                # Shape: (n_preprocessed,) - excludes label-to-label self-attention
                label_attn_to_features = raw_attn[-1, :-1]
                layer_attentions.append(label_attn_to_features)

        assert layer_attentions, "No attention maps found."

        # Average across layers
        avg_attn_to_label = np.mean(layer_attentions, axis=0)
        assert avg_attn_to_label.shape == (mapping_info["n_preprocessed"],), (
            f"Got label attention for {avg_attn_to_label.shape[0]} tokens, "
            f"expected {mapping_info['n_preprocessed']} feature tokens."
        )

        # Mean over the tokens derived from each input feature; NaN for dropped ones.
        result = np.full(self.n_features_in_, np.nan)
        for feature, positions in mapping_info["original_to_preprocessed"].items():
            if positions:
                result[feature] = avg_attn_to_label[positions].mean()

        return result
