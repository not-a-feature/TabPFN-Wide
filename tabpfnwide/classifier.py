import torch
import numpy as np
from scipy.sparse import csr_matrix, diags
import os
import warnings
import pathlib
import dataclasses
from tabpfn import TabPFNClassifier
from tabpfn.model.loading import load_model_criterion_config
from tabpfnwide.patches import fit as patched_fit
from tabpfn.base import determine_precision
from tabpfn.utils import infer_random_state, infer_devices, update_encoder_params

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

        if save_attention_maps and (n_estimators != 1 or features_per_group != 1):
            raise ValueError(
                "save_attention_maps can only be True when n_estimators=1 and features_per_group=1"
            )

        # Initialize parent TabPFNClassifier
        # We pass ignore_pretraining_limits=True by default, but allow override
        if "ignore_pretraining_limits" not in kwargs:
            kwargs["ignore_pretraining_limits"] = True

        super().__init__(
            device=device,
            n_estimators=n_estimators,
            model_path=model_path,
            **kwargs,
        )

        # Restore model_path after super().__init__ overwrites it with default "auto"
        self.model_path = model_path
        self.model_name = model_name
        self.features_per_group = features_per_group
        self.n_estimators = n_estimators
        self.save_attention_maps = save_attention_maps
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self._load_model()

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

    def _initialize_model_variables(self):
        # We already loaded the model in __init__
        self.models_ = [self.model]

        static_seed, rng = infer_random_state(self.random_state)

        self.devices_ = infer_devices(self.device)

        (
            self.use_autocast_,
            self.forced_inference_dtype_,
            byte_size,
        ) = determine_precision(self.inference_precision, self.devices_)

        # Handle inference config override
        if hasattr(self, "inference_config_"):
            self.inference_config_ = self.inference_config_.override_with_user_input(
                user_config=self.inference_config
            )

            outlier_removal_std = self.inference_config_.OUTLIER_REMOVAL_STD
            if outlier_removal_std == "auto":
                outlier_removal_std = (
                    self.inference_config_._CLASSIFICATION_DEFAULT_OUTLIER_REMOVAL_STD
                )
        else:
            # Fallback if inference_config_ was not set (should not happen with updated _load_wide_model)
            outlier_removal_std = None

        update_encoder_params(
            models=self.models_,
            remove_outliers_std=outlier_removal_std,
            seed=static_seed,
            differentiable_input=self.differentiable_input,
        )

        return byte_size, rng

    def _load_model(self):
        # Load the base model structure
        models, _, configs, inference_config = load_model_criterion_config(
            model_path=None,
            check_bar_distribution_criterion=False,
            cache_trainset_representation=False,
            which="classifier",
            version="v2",
            download_if_not_exists=True,
        )
        model = models[0]

        self.configs_ = configs
        self.inference_config_ = inference_config

        model.features_per_group = self.features_per_group
        self.configs_[0].features_per_group = self.features_per_group

        if self.model_name != "v2":
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

            # Handle DDP-wrapped checkpoints and direct state dicts
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            model.load_state_dict(state_dict)

        return model

    def fit(self, X, y):
        # Store n_features_in_ for attention map cropping
        self.n_features_in_ = X.shape[1]

        if self.save_attention_maps:
            for layer in self.model.transformer_encoder.layers:
                if hasattr(layer, "self_attn_between_features"):
                    layer.self_attn_between_features.save_att_map = True
                    layer.self_attn_between_features.number_of_samples = X.shape[0]
                    layer.self_attn_between_features.attention_map = None

        # Use the patched fit function, passing the loaded model
        return patched_fit(self, X, y, model=self.model)

    def get_attention_maps(self):
        """Return attention maps mapped to original input features.

        This method handles the preprocessing transformations by:
        1. Extracting the raw attention maps from the transformer
        2. Using the preprocessor's index_permutation_ to map back to original features
        3. Handling feature doubling from append_to_original=True
        4. Aggregating attention for each original feature

        Returns:
            List of numpy arrays, one per transformer layer, with shape
            (n_features_in_, n_features_in_) representing attention between
            original input features.
        """
        if not self.save_attention_maps:
            raise ValueError("Attention maps were not saved during training.")

        raw_maps = []
        for layer in self.model.transformer_encoder.layers:
            if hasattr(layer, "self_attn_between_features"):
                attn = getattr(layer.self_attn_between_features, "attention_map", None)
                if attn is not None:
                    raw_maps.append(attn.numpy())

        n_original = self.n_features_in_

        # Get the preprocessing mapping info
        mapping_info = self._get_feature_mapping_info()
        if mapping_info is None:
            warnings.warn("Could not get feature mapping info. Returning None.")
            return None

        original_to_preprocessed = mapping_info["original_to_preprocessed"]
        n_preprocessed = mapping_info["n_preprocessed"]

        # Build sparse mapping matrix P
        # P[i, p] = 1 if original feature i maps to preprocessed feature p
        rows = []
        cols = []
        data = []

        for orig_idx, positions in original_to_preprocessed.items():
            for pos in positions:
                rows.append(orig_idx)
                cols.append(pos)
                data.append(1.0)

        # Shape is (n_original, n_preprocessed_max)
        # We use a sufficiently large shape to cover all indices found
        max_col = max(cols) if cols else 0
        n_matrix_cols = max(n_preprocessed, max_col + 1)

        P = csr_matrix((data, (rows, cols)), shape=(n_original, n_matrix_cols))

        mapped_maps = []
        for raw_attn in raw_maps:
            curr_dim = raw_attn.shape[0]

            # Match P columns to raw_attn dimensions
            if P.shape[1] > curr_dim:
                P_sliced = P[:, :curr_dim]
            elif P.shape[1] < curr_dim:
                # Slice raw_attn to match P columns
                raw_attn = raw_attn[: P.shape[1], : P.shape[1]]
                P_sliced = P
            else:
                P_sliced = P

            # Calculate normalization factors (counts of valid mappings)
            row_sums = np.array(P_sliced.sum(axis=1)).flatten()
            row_sums[row_sums == 0] = 1.0

            # Create normalized mapping matrix
            scale = 1.0 / row_sums
            D = diags(scale)
            P_norm = D @ P_sliced

            # Compute mapped attention: P_norm @ raw_attn @ P_norm.T
            temp = P_norm @ raw_attn  # (n_original, curr_dim)
            mapped_attn = temp @ P_norm.T  # (n_original, n_original)

            # Convert to dense array if needed
            if hasattr(mapped_attn, "toarray"):
                mapped_attn = mapped_attn.toarray()

            mapped_maps.append(mapped_attn)

        return mapped_maps

    def _get_feature_mapping_info(self):
        """Extract feature mapping from preprocessor to original features.

        Returns a dict with:
        - original_to_preprocessed: dict mapping original feature idx to list
          of preprocessed positions
        - n_preprocessed: total number of preprocessed features
        - index_permutation: the shuffle permutation applied

        Returns None if mapping cannot be extracted.
        """
        # Need the executor to access preprocessors
        if not hasattr(self, "executor_"):
            return None

        # Get preprocessor from executor
        preprocessor = None
        if hasattr(self.executor_, "preprocessors") and self.executor_.preprocessors:
            preprocessor = self.executor_.preprocessors[0]
        elif hasattr(self.executor_, "preprocessor"):
            preprocessor = self.executor_.preprocessor

        if preprocessor is None:
            return None

        # Extract info from preprocessor steps
        n_original = self.n_features_in_
        index_permutation = None
        append_to_original = False

        # Walk through preprocessor steps
        steps = getattr(preprocessor, "steps", [])
        for step in steps:
            step_obj = step[1] if isinstance(step, tuple) else step

            # Check for ShuffleFeaturesStep
            if hasattr(step_obj, "index_permutation_"):
                index_permutation = step_obj.index_permutation_

            # Check for ReshapeFeatureDistributionsStep
            if hasattr(step_obj, "append_to_original"):
                append_to_original = step_obj.append_to_original

        # Build the mapping from original feature idx to preprocessed positions
        original_to_preprocessed = {}

        if index_permutation is not None:
            # Convert to list if tensor
            if hasattr(index_permutation, "tolist"):
                perm = index_permutation.tolist()
            else:
                perm = list(index_permutation)

            n_preprocessed = len(perm)

            # When append_to_original=True, positions 0..n-1 are originals,
            # positions n..2n-1 are transformed versions
            if append_to_original:
                for orig_idx in range(n_original):
                    positions = []
                    # Find where this original feature ended up after shuffle
                    for new_pos, old_pos in enumerate(perm):
                        # Original features are at indices 0..n-1 before shuffle
                        # Transformed versions are at indices n..2n-1
                        if old_pos == orig_idx or old_pos == (orig_idx + n_original):
                            positions.append(new_pos)
                    original_to_preprocessed[orig_idx] = positions if positions else [orig_idx]
            else:
                # No feature doubling - direct mapping through permutation
                for orig_idx in range(n_original):
                    positions = []
                    for new_pos, old_pos in enumerate(perm):
                        if old_pos == orig_idx:
                            positions.append(new_pos)
                    original_to_preprocessed[orig_idx] = positions if positions else [orig_idx]
        else:
            # No shuffle info - assume identity mapping
            n_preprocessed = n_original
            for i in range(n_original):
                original_to_preprocessed[i] = [i]

        return {
            "original_to_preprocessed": original_to_preprocessed,
            "n_preprocessed": n_preprocessed,
            "index_permutation": index_permutation,
            "append_to_original": append_to_original,
        }

    def get_raw_attention_maps(self):
        """Return raw attention maps without any processing.

        Returns a tuple of (maps, n_features_in) where maps is a list of raw
        attention matrices and n_features_in is the number of input features.
        """
        if not self.save_attention_maps:
            raise ValueError("Attention maps were not saved during training.")

        maps = []
        for layer in self.model.transformer_encoder.layers:
            if hasattr(layer, "self_attn_between_features"):
                attn = getattr(layer.self_attn_between_features, "attention_map", None)
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

    def get_attention_to_label(self):
        """Return attention scores showing how much the label attends to each feature.

        In TabPFN's architecture, the label (y) is concatenated as the last token
        in the feature dimension. This method extracts how much the label token
        (as a query) attends to each feature (as keys), indicating feature
        importance for the prediction.

        Returns:
            numpy array of shape (n_features_in_,) where each value represents
            how much the label token attends to that original input feature,
            averaged across all transformer layers.
        """
        if not self.save_attention_maps:
            raise ValueError("Attention maps were not saved during training.")

        n_original = self.n_features_in_

        # Get the preprocessing mapping info
        mapping_info = self._get_feature_mapping_info()
        assert mapping_info is not None, "Could not get feature mapping info."

        original_to_preprocessed = mapping_info["original_to_preprocessed"]

        # Collect attention from label to features from each layer
        layer_attentions = []
        for layer in self.model.transformer_encoder.layers:
            if hasattr(layer, "self_attn_between_features"):
                attn = getattr(layer.self_attn_between_features, "attention_map", None)
                if attn is not None:
                    raw_attn = attn.numpy()
                    # Last ROW = how much the label (query) attends to each feature (key)
                    # Shape: (n_preprocessed,) - excludes label-to-label self-attention
                    label_attn_to_features = raw_attn[-1, :-1]
                    layer_attentions.append(label_attn_to_features)

        assert layer_attentions, "No attention maps found."

        # Average across layers
        avg_attn_to_label = np.mean(layer_attentions, axis=0)

        # Map preprocessed positions back to original features
        # Each original feature may map to multiple preprocessed positions
        result = np.zeros(n_original)
        for orig_idx in range(n_original):
            positions = original_to_preprocessed.get(orig_idx, [orig_idx])
            # Filter positions that are within bounds
            valid_positions = [p for p in positions if p < len(avg_attn_to_label)]
            if valid_positions:
                result[orig_idx] = np.mean([avg_attn_to_label[p] for p in valid_positions])

        return result
