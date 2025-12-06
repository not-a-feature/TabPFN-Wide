# TabPFN Migration Guide: v2.0 to v2.5

This document outlines the changes needed to migrate from TabPFN v2.1.1 (v2.0 series) to TabPFN v6.0.6 (v2.5 series) based on the diff analysis.

## Overview

The migration from TabPFN v2.0 to v2.5 involves several breaking changes in the package structure, model configuration, and API. The v2.5 release introduces significant improvements in model capacity and performance but requires code updates.

## Key Changes Summary

### 1. **Package Version Change**
- **Old**: `version = "2.1.1"`
- **New**: `version = "6.0.6"`

### 2. **Model Architecture Changes**

#### Model Configuration Parameters

Several model configuration parameters have been added or modified:

**New Parameters:**
- `encoder_type`: Type of input encoder (`"linear"` or `"mlp"`)
- `encoder_mlp_hidden_dim`: Hidden dimension for MLP encoder (default: 1024)
- `encoder_mlp_num_layers`: Number of layers in MLP encoder (default: 2)
- `num_thinking_rows`: Number of "thinking rows" prepended to datasets (default: 0)

**Modified Parameters:**
- `features_per_group`: Changed from `Literal[1, 2]` to `PositiveInt` (more flexible)
- `multiquery_item_attention_for_test_set`: Changed from `Literal[True]` to `bool`

**Removed Constraints:**
- The `recompute_layer` docstring was simplified (removed mention of `force_recompute_layer` argument)

#### New Encoder: MLPInputEncoderStep

A new MLP-based input encoder has been added alongside the existing linear encoder:

```python
class MLPInputEncoderStep(SeqEncStep):
    """An MLP-based input encoder step."""
    
    def __init__(
        self,
        *,
        num_features: int,
        emsize: int,
        hidden_dim: int | None = None,
        activation: str = "gelu",
        num_layers: int = 2,
        replace_nan_by_zero: bool = False,
        bias: bool = True,
        in_keys: tuple[str, ...] = ("main",),
        out_keys: tuple[str, ...] = ("output",),
    ):
        # ...
```

### 3. **Attention Mechanism Changes**

#### GQA (Grouped Query Attention) Support

The attention mechanism now includes better support for Grouped Query Attention:

**Old approach:**
- Simple check for PyTorch version and CUDA availability
- Static module-level `USE_TORCH_2_GQA` flag

**New approach:**
- Dynamic GQA support detection using `contextlib.suppress`
- Runtime check for `enable_gqa` parameter support
- Improved handling of multi-GPU settings

**Key code change:**
```python
# Old: Static check at module level
USE_TORCH_2_GQA = _gqa_is_supported()

# New: Dynamic check during computation
with contextlib.suppress(TypeError, RuntimeError):
    _ = torch.nn.functional.scaled_dot_product_attention(
        torch.empty(1, 1, 1, 1),
        torch.empty(1, 1, 1, 1),
        torch.empty(1, 1, 1, 1),
        enable_gqa=True,
    )
USE_TORCH_2_GQA = True
```

#### Chunked Attention Removed

The `scaled_dot_product_attention_chunked` method was removed. This was a workaround for batch size limitations (>65,535) in CUDA kernels.

### 4. **Dependency Changes**

#### Updated Dependencies:

```toml
# Old
"scikit-learn>=1.2.0,<1.7"
"huggingface-hub>=0.0.1,<1"

# New
"scikit-learn>=1.2.0,<1.8"
"huggingface-hub>=0.19.0,<2"
```

#### New Dependencies:

```toml
"numpy>=1.21.6,<3"
"joblib>=1.2.0"
"tabpfn-common-utils[telemetry-interactive]>=0.2.7"
"pyobjc-framework-Metal; sys_platform == 'darwin' and python_version > '3.9'"
"kditransform>=1.2"
```

### 5. **Model Interface Changes**

#### Finetuning API

**Old:**
```python
model = classifier.models_[0]
optimizer = Adam(model.parameters(), lr=learning_rate)
```

**New:**
```python
# Single model access via model_ (not models_[0])
optimizer = Adam(classifier.model_.parameters(), lr=learning_rate)
```

**Note:** The new API enforces single model usage for finetuning and raises an error if multiple models are configured.

#### Regressor Changes

**Old variable names:**
```python
X_trains_preprocessed, X_tests_preprocessed, y_trains_znorm, y_test_znorm
raw_space_bardist_, znorm_space_bardist_
regressor.raw_space_bardist_ = raw_space_bardist_[0]
regressor.bardist_ = znorm_space_bardist_[0]
```

**New variable names:**
```python
X_trains_p, X_tests_p, y_trains_p, y_test_std
norm_bardist, bardist
regressor.normalized_bardist_ = norm_bardist[0]
```

### 6. **New Features**

#### KV Cache Support

New `fit_mode="fit_with_cache"` option for faster prediction:

```python
clf = TabPFNClassifier(fit_mode="fit_with_cache")
clf.fit(X_train, y_train)
# Predictions are now faster due to cached key-value pairs
predictions = clf.predict(X_test)
```

#### Tuning Configuration

New tuning capabilities for probability calibration and decision threshold optimization:

```python
clf = TabPFNClassifier(
    eval_metric="f1",
    tuning_config={"tune_decision_thresholds": True},
)
clf.fit(X_train, y_train)
```

### 7. **BarDistribution Changes**

**Removed method:**
```python
def has_equal_borders(self, other: BarDistribution) -> bool:
    """Check if two BarDistributions have equal borders."""
    return torch.equal(self.borders, other.borders)
```

**Modified code patterns:**
```python
# Old
ys = ys.repeat((*logits.shape[:-1], 1))

# New
ys = ys.repeat(logits.shape[:-1] + (1,))
```

### 8. **Preprocessing Changes**

#### RemoveEmptyFeaturesEncoderStep

This encoder step was modified to be a no-op (does nothing) but is kept for backward compatibility with saved models:

```python
class RemoveEmptyFeaturesEncoderStep(SeqEncStep):
    """Encoder step to remove empty (constant) features.
    Was changed to NOT DO ANYTHING, the removal of empty features now
    done elsewhere, but the saved model still needs this encoder step.
    TODO: REMOVE.
    """
```

**Variable rename:**
- `column_selection_mask` → `sel`

#### select_features Function

**Old:**
```python
# Do nothing if we need to select all of the features
if torch.all(sel):
    return x

new_x = x.detach().clone()
# ... modify in place
```

**New:**
```python
# Always create new tensor
new_x = torch.zeros(
    (sequence_length, B, total_features),
    device=x.device,
    dtype=x.dtype,
)
# ... fill with selected features
```

### 9. **Telemetry**

TabPFN v2.5 introduces anonymous telemetry collection (can be opted out):

```bash
export TABPFN_DISABLE_TELEMETRY=1
```

Telemetry includes:
- Python version
- TabPFN version
- Rounded dataset dimensions
- Fit/predict duration
- Task type (classification/regression)

### 10. **Documentation and Examples**

**Removed examples:**
- `kv_cache_fast_prediction.py` (removed from v2.5)
- `tabpfn_with_tuning.py` (removed from v2.5)

**Modified examples:**
- `finetune_classifier.py`: Updated to use `model_` instead of `models_[0]`
- `finetune_regressor.py`: Updated variable names and API

## Migration Checklist

### For Your Codebase

1. **Update model loading code:**
   - [ ] If using finetuning, change `classifier.models_[0]` to `classifier.model_`
   - [ ] Remove checks for multiple models (now enforced by the API)

2. **Update regressor code:**
   - [ ] Rename `raw_space_bardist_` to `normalized_bardist_`
   - [ ] Update variable names in preprocessing pipeline
   - [ ] Change `bardist_` to `normalized_bardist_`

3. **Update model configuration:**
   - [ ] Review if you need the new `encoder_type` parameter (default: `"linear"`)
   - [ ] Consider using `num_thinking_rows` if applicable
   - [ ] Update `features_per_group` if you were relying on the `Literal[1, 2]` constraint

4. **Update dependencies:**
   - [ ] Update `pyproject.toml` or `requirements.txt` with new dependency versions
   - [ ] Add new dependencies: `joblib`, `tabpfn-common-utils`, `kditransform`
   - [ ] Update scikit-learn to `<1.8` (from `<1.7`)
   - [ ] Update huggingface-hub to `>=0.19.0,<2` (from `>=0.0.1,<1`)

5. **Test attention mechanisms:**
   - [ ] Verify GQA support on your hardware (requires CUDA compute capability >= 8.0)
   - [ ] Test with your typical batch sizes (chunked attention was removed)

6. **Consider new features:**
   - [ ] Evaluate if KV cache (`fit_mode="fit_with_cache"`) would benefit your use case
   - [ ] Consider using tuning configuration for better performance

7. **Handle telemetry:**
   - [ ] Decide if you want to opt out of telemetry
   - [ ] Set `TABPFN_DISABLE_TELEMETRY=1` if needed

## Potential Breaking Changes

### High Priority

1. **Finetuning API change**: `models_[0]` → `model_` (will cause AttributeError)
2. **Regressor variable names**: Multiple renames in preprocessing pipeline
3. **Dependency version bumps**: May cause compatibility issues

### Medium Priority

4. **Attention chunking removed**: May cause issues with very large batch sizes (>65k)
5. **Model configuration changes**: If you're creating models programmatically
6. **BarDistribution API**: `has_equal_borders` method removed

### Low Priority

7. **select_features behavior**: Now always creates new tensor instead of in-place modification
8. **RemoveEmptyFeaturesEncoderStep**: Now a no-op (should be transparent)

## Verification Steps

After migration:

1. **Run existing tests**: Ensure all unit tests pass
2. **Test model loading**: Verify that saved models from v2.0 can be loaded in v2.5
3. **Test finetuning**: If you use finetuning, verify the new API works
4. **Performance testing**: Compare inference times and memory usage
5. **Accuracy validation**: Ensure model predictions are consistent

## Additional Resources

- [TabPFN v2.5 Model Report](https://priorlabs.ai/technical-reports/tabpfn-2-5-model-report)
- [TabPFN v2.5 License](https://huggingface.co/Prior-Labs/tabpfn_2_5/blob/main/LICENSE)
- [Telemetry Documentation](https://github.com/PriorLabs/TabPFN/blob/main/TELEMETRY.md)

## Notes

- TabPFN v2.5 is released under a different license (TABPFN-2.5 Non-Commercial License v1.0)
- The v2.5 model scales to datasets with up to 50,000 samples and 2,000 features (vs 10,000 samples in v2.0)
- To continue using v2.0 models: `TabPFNClassifier.create_default_for_version(ModelVersion.V2)`
