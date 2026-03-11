import os
import sys
import numpy as np
import traceback
import torch
import warnings
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tabpfnwide.classifier import TabPFNWideClassifier, VALID_MODELS


def get_available_models():
    package_dir = os.path.dirname(os.path.abspath(sys.modules["tabpfnwide.classifier"].__file__))
    models_dir = os.path.join(package_dir, "models")

    available = ["v2"]
    for m in VALID_MODELS:
        if m == "v2":
            continue
        expected_path = os.path.join(models_dir, f"tabpfn-{m}.pt")
        if os.path.exists(expected_path):
            available.append(m)
        else:
            print(f"Skipping {m}, file not found")
    return available


def test_model_loading_and_config():
    available_models = get_available_models()
    print(f"Testing basic loading for: {available_models}")

    X, y = make_classification(n_samples=20, n_features=4, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for model_name in available_models:
        print(f"  Testing {model_name}...")
        try:
            # Test default config
            clf = TabPFNWideClassifier(
                model_name=model_name,
                device="cpu",
            )
            clf.fit(X_train, y_train)
            clf.predict(X_test)
            print(f"    Pass: {model_name} basic fit/predict")
        except Exception:
            print(f"    Fail: {model_name}")
            traceback.print_exc()
            return False
    return True


def run_single_attention_test(categorical=False):
    available_models = get_available_models()
    if not available_models:
        print("No models available to test.")
        return False

    overall_pass = True
    mode_str = "CATEGORICAL Test" if categorical else "NUMERICAL Test"
    print(f"\n=== Testing Attention Map Correctness ({mode_str}) ===")

    # 1. Create synthetic dataset
    # 5 informative features
    n_features = 95
    n_informative = 5
    n_samples = 300
    check_top_k = 10
    # Generate ordered data first to identify informative features easily
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        shuffle=False,
        random_state=123456,
    )

    # Manually shuffle features to track indices
    rng = np.random.RandomState(123456)
    p = rng.permutation(n_features)
    X_shuffled = X[:, p]

    # Identify which columns in X_shuffled are the informative ones
    # The original informative features were at indices 0 to n_informative-1
    # X_shuffled[:, j] comes from X[:, p[j]]
    # So if p[j] < n_informative, then column j is an informative feature
    informative_idxs = np.where(p < n_informative)[0]
    noise_idxs = np.where(p >= n_informative)[0]

    # Handle Categorical conversion
    if categorical:
        import pandas as pd

        X_df = pd.DataFrame(X_shuffled, columns=[f"feat_{i}" for i in range(n_features)])

        # Convert a subset of informative and noise features to categorical (binned strings)
        # Convert first 2 informative and first 20 noise features
        cats_to_convert = np.concatenate([informative_idxs[:2], noise_idxs[:20]])

        for idx in cats_to_convert:
            col_name = X_df.columns[idx]
            # Simple binning to create categories
            X_df[col_name] = pd.qcut(X_df[col_name], q=10, duplicates="drop").astype(str)

        X_input = X_df
    else:
        X_input = X_shuffled

    for model_name in available_models:
        print(f"\n--- Testing Attention Map: {model_name} [{mode_str}] ---")
        try:
            # We use n_estimators=1, features_per_group=1 as required for save_attention_maps=True
            clf = TabPFNWideClassifier(
                model_name=model_name,
                device="cpu",
                n_estimators=1,
                features_per_group=1,
                save_attention_maps=True,
            )

            # Test using shuffled data but tracking importance via indices
            clf.fit(X_input, y)
            clf.predict(X_input)
            maps = clf.get_attention_maps()

            if not maps:
                print(f"FAILURE ({model_name}): No attention maps returned.")
                overall_pass = False
                continue

            avg_map = np.mean(maps, axis=0)
            if avg_map.shape != (n_features, n_features):
                print(
                    f"FAILURE ({model_name}): Expected map shape ({n_features}, {n_features}), got {avg_map.shape}"
                )
                overall_pass = False
                continue

            # Calculate Importance
            importance = avg_map.sum(axis=0)

            informative_importance = importance[informative_idxs].mean()
            noise_importance = importance[noise_idxs].mean()

            print(f"  Mean importance (Informative): {informative_importance:.4f}")
            print(f"  Mean importance (Noise):       {noise_importance:.4f}")

            if informative_importance > noise_importance:
                print(
                    f"  SUCCESS ({model_name}): Informative features received more attention (Weak Test)."
                )
            else:
                print(
                    f"  FAILURE ({model_name}): Noise features received more (or equal) attention (Weak Test)."
                )
                overall_pass = False

            # Strong Test
            top_indices = np.argsort(importance)[::-1][:check_top_k]

            # Use sets for order-independent comparison
            # We want to check if the informative features are captured in the top K (where K > n_informative)
            if set(informative_idxs).issubset(set(top_indices)):
                print(
                    f"  SUCCESS ({model_name}): Informative indices are contained in Top {n_informative*2} importance scores (Strong Test)."
                )
            else:
                print(
                    f"  FAILURE ({model_name}): Top {n_informative} features DO NOT match exactly (Strong Test)."
                )
                print(f"    Expected: {set(informative_idxs.tolist())}")
                print(f"    Got:      {set(top_indices.tolist())}")
                overall_pass = False

        except Exception:
            print(f"  ERROR ({model_name}): Exception occurred.")
            traceback.print_exc()
            overall_pass = False

    return overall_pass


def test_attention_to_label():
    """Test that get_attention_to_label() correctly identifies informative features."""
    available_models = get_available_models()
    if not available_models:
        print("No models available to test.")
        return False

    print("\n=== Testing Attention to Label ===")
    overall_pass = True

    n_features = 50
    n_informative = 5
    n_samples = 200

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        shuffle=False,
        random_state=42,
    )

    # Shuffle features and track indices
    rng = np.random.RandomState(42)
    perm = rng.permutation(n_features)
    X_shuffled = X[:, perm]

    informative_idxs = np.where(perm < n_informative)[0]
    noise_idxs = np.where(perm >= n_informative)[0]

    for model_name in available_models:
        print(f"\n--- Testing Attention to Label: {model_name} ---")
        try:
            clf = TabPFNWideClassifier(
                model_name=model_name,
                device="cpu",
                n_estimators=1,
                features_per_group=1,
                save_attention_maps=True,
            )

            clf.fit(X_shuffled, y)
            clf.predict(X_shuffled)

            attn_to_label = clf.get_attention_to_label()

            # Verify shape
            if attn_to_label.shape != (n_features,):
                print(
                    f"  FAILURE ({model_name}): Expected shape ({n_features},), got {attn_to_label.shape}"
                )
                overall_pass = False
                continue

            informative_attn = attn_to_label[informative_idxs].mean()
            noise_attn = attn_to_label[noise_idxs].mean()

            print(f"  Mean attention to label (Informative): {informative_attn:.4f}")
            print(f"  Mean attention to label (Noise):       {noise_attn:.4f}")

            if informative_attn > noise_attn:
                print(
                    f"  SUCCESS ({model_name}): Informative features attend more to label (Weak Test)."
                )
            else:
                print(
                    f"  FAILURE ({model_name}): Noise features attend more (or equal) to label (Weak Test)."
                )
                overall_pass = False

            # Strong Test: Check if informative features are in top-k
            check_top_k = n_informative * 2
            top_indices = np.argsort(attn_to_label)[::-1][:check_top_k]

            if set(informative_idxs).issubset(set(top_indices)):
                print(
                    f"  SUCCESS ({model_name}): All informative features in top {check_top_k} (Strong Test)."
                )
            else:
                print(
                    f"  FAILURE ({model_name}): Not all informative features in top {check_top_k} (Strong Test)."
                )
                print(f"    Expected: {set(informative_idxs.tolist())}")
                print(f"    Got top {check_top_k}: {set(top_indices.tolist())}")
                overall_pass = False

        except Exception:
            print(f"  ERROR ({model_name}): Exception occurred.")
            traceback.print_exc()
            overall_pass = False

    return overall_pass


def test_attention_maps():
    print("\n=== Running Consolidated Attention Map Tests ===")
    # 3. Run attention-to-label test
    pass_attn_to_label = test_attention_to_label()

    # 2. Run categorical test
    pass_categorical = run_single_attention_test(categorical=True)

    # 1. Run standard numerical test
    pass_numerical = run_single_attention_test(categorical=False)

    return pass_numerical and pass_categorical and pass_attn_to_label


def main():
    success = True
    if not test_model_loading_and_config():
        success = False

    if not test_attention_maps():
        success = False

    if success:
        print("\nALL CONSOLIDATED TESTS PASSED")
    else:
        print("\nSOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
