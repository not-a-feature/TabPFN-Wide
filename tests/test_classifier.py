"""Pytest tests for TabPFNWideClassifier argument validation and basic behaviour.

The validation tests are cheap (no model load). The end-to-end v2 test downloads
the base TabPFN-v2 weights and is marked ``slow`` so it can be skipped with
``pytest -m "not slow"``.
"""
from __future__ import annotations

import functools
import os

import numpy as np
import pytest
import torch
from scipy.stats import spearmanr
from sklearn.datasets import make_classification

from tabpfnwide.classifier import VALID_MODELS, TabPFNWideClassifier
from tabpfnwide.patches import forward_recording_attention


# ---------------------------------------------------------------------------
# Argument validation — no model load required
# ---------------------------------------------------------------------------


def test_neither_name_nor_path_raises():
    with pytest.raises(ValueError, match="Either model_name or model_path"):
        TabPFNWideClassifier(model_name="", model_path="", device="cpu")


def test_both_name_and_path_raises(tmp_path):
    fake = tmp_path / "fake.pt"
    fake.write_bytes(b"\x00")
    with pytest.raises(ValueError, match="Either model_name or model_path"):
        TabPFNWideClassifier(model_name="v2", model_path=str(fake), device="cpu")


def test_unknown_model_name_raises():
    with pytest.raises(ValueError, match="not recognized"):
        TabPFNWideClassifier(model_name="not-a-real-model", device="cpu")


def test_unknown_model_name_lists_valid_models():
    with pytest.raises(ValueError) as exc:
        TabPFNWideClassifier(model_name="nope", device="cpu")
    msg = str(exc.value)
    for name in VALID_MODELS:
        assert name in msg


def test_nonexistent_model_path_raises(tmp_path):
    missing = tmp_path / "does_not_exist.pt"
    with pytest.raises(ValueError, match="does not exist"):
        TabPFNWideClassifier(model_path=str(missing), device="cpu")


def test_save_attention_maps_rejects_multiple_estimators():
    with pytest.raises(ValueError, match="save_attention_maps"):
        TabPFNWideClassifier(
            model_name="v2",
            device="cpu",
            n_estimators=2,
            features_per_group=1,
            save_attention_maps=True,
        )


def test_save_attention_maps_rejects_unknown_mode():
    with pytest.raises(ValueError, match="save_attention_maps must be"):
        TabPFNWideClassifier(model_name="v2", device="cpu", save_attention_maps="full")


def test_save_attention_maps_rejects_features_per_group_gt_1():
    with pytest.raises(ValueError, match="save_attention_maps"):
        TabPFNWideClassifier(
            model_name="v2",
            device="cpu",
            n_estimators=1,
            features_per_group=2,
            save_attention_maps=True,
        )


# ---------------------------------------------------------------------------
# Fail-fast in _build_model_specs (the regression this PR fixes)
# ---------------------------------------------------------------------------


def test_build_model_specs_rejects_none_path_for_wide_model():
    """Calling the helper directly with a wide model name but no path must
    raise a precise ValueError instead of letting torch.load see ``None``."""
    with pytest.raises(ValueError, match="must point to an existing file"):
        TabPFNWideClassifier._build_model_specs(
            model_name="wide-v2-1.5k",
            model_path=None,
            features_per_group=1,
            device="cpu",
        )


def test_build_model_specs_rejects_empty_path_for_wide_model():
    with pytest.raises(ValueError, match="must point to an existing file"):
        TabPFNWideClassifier._build_model_specs(
            model_name="wide-v2-1.5k",
            model_path="",
            features_per_group=1,
            device="cpu",
        )


def test_build_model_specs_rejects_nonexistent_path_for_wide_model(tmp_path):
    missing = tmp_path / "ghost.pt"
    with pytest.raises(ValueError, match="must point to an existing file"):
        TabPFNWideClassifier._build_model_specs(
            model_name="wide-v2-1.5k",
            model_path=str(missing),
            features_per_group=1,
            device="cpu",
        )


# ---------------------------------------------------------------------------
# End-to-end v2 sanity test — downloads the base TabPFN-v2 weights
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_v2_fit_predict_smoke():
    X, y = make_classification(n_samples=40, n_features=8, random_state=0)
    Xtr, Xte = X[:30], X[30:]
    ytr, yte = y[:30], y[30:]
    clf = TabPFNWideClassifier(model_name="v2", device="cpu", n_estimators=2)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    proba = clf.predict_proba(Xte)
    assert pred.shape == yte.shape
    assert proba.shape == (len(yte), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Local wide checkpoint round-trip — only runs if a checkpoint is present
# ---------------------------------------------------------------------------


def _local_wide_checkpoints():
    pkg_models = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "tabpfnwide", "models"
    )
    found = []
    for name in VALID_MODELS:
        if name == "v2":
            continue
        path = os.path.join(pkg_models, f"tabpfn-{name}.pt")
        if os.path.isfile(path):
            found.append((name, path))
    return found


@pytest.mark.slow
@pytest.mark.parametrize("name,path", _local_wide_checkpoints())
def test_local_wide_checkpoint_fit_predict(name, path):
    X, y = make_classification(n_samples=30, n_features=8, random_state=0)
    Xtr, Xte = X[:22], X[22:]
    ytr, _ = y[:22], y[22:]
    clf = TabPFNWideClassifier(model_path=path, device="cpu")
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    assert pred.shape == (len(Xte),)


# ---------------------------------------------------------------------------
# Attention-recording smoke test — guards the instance-level patching in
# __init__/_predict_proba against regressions (previously a global
# import-time monkeypatch on MultiHeadAttention._compute).
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_save_attention_maps_records_and_resets():
    n_features = 8
    X, y = make_classification(n_samples=40, n_features=n_features, random_state=0)
    Xtr, Xte = X[:30], X[30:]
    ytr = y[:30]

    clf = TabPFNWideClassifier(
        model_name="v2",
        device="cpu",
        n_estimators=1,
        features_per_group=1,
        save_attention_maps=True,
        random_state=0,
    )
    clf.fit(Xtr, ytr)

    # Sanity: the recording forward is installed on every between-features
    # attention module.
    patched_modules = [
        block.per_sample_attention_between_features for block in clf._wide_model.blocks
    ]
    assert patched_modules, "no between-features attention modules found"
    assert all(m.forward.__func__ is forward_recording_attention for m in patched_modules)

    clf.predict_proba(Xte)

    raw_maps, n_in = clf.get_raw_attention_maps()
    assert n_in == n_features
    assert len(raw_maps) >= 1, "expected at least one layer of raw attention maps"
    for m in raw_maps:
        assert m.ndim == 2 and m.shape[0] == m.shape[1], (
            f"raw attention map must be square, got shape {m.shape}"
        )

    mapped = clf.get_attention_maps()
    assert mapped is not None and len(mapped) >= 1
    for m in mapped:
        assert m.shape == (n_features, n_features), (
            f"mapped attention map must be (n_features, n_features), got {m.shape}"
        )

    attn_to_label = clf.get_attention_to_label()
    assert attn_to_label.shape == (n_features,)

    # Reset check: a second predict on the same data must not roughly double
    # the recorded scale. Without _predict_proba's reset, attention_map would
    # accumulate and the magnitudes would be ~2x the first call.
    first_scale = np.mean([np.abs(m).sum() for m in raw_maps])
    clf.predict_proba(Xte)
    raw_maps_2, _ = clf.get_raw_attention_maps()
    second_scale = np.mean([np.abs(m).sum() for m in raw_maps_2])
    ratio = second_scale / first_scale
    assert 0.9 < ratio < 1.1, (
        f"attention buffers were not reset between predicts: "
        f"second/first scale ratio = {ratio:.3f} (expected ~1.0)"
    )


# ---------------------------------------------------------------------------
# Label-only attention recording. The full maps hold, per layer, the sum over
# rows divided by the number of training rows, so their label row rescaled by
# n_train / n_rows must equal the per-row mean recorded in label mode.
# ---------------------------------------------------------------------------

PKG_MODELS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tabpfnwide", "models")

LABEL_CASES = {
    "v2-binary": dict(model="v2", n_train=30, n_test=10, n_features=12, n_classes=2, nan=0.0),
    "1.5k-3class-200feat": dict(
        model="wide-v2-1.5k", n_train=45, n_test=15, n_features=200, n_classes=3, nan=0.0
    ),
    "5k-nan10": dict(model="wide-v2-5k", n_train=40, n_test=12, n_features=60, n_classes=2, nan=0.1),
}


def _classifier(model, mode):
    if model == "v2":
        source = dict(model_name="v2")
    else:
        source = dict(model_path=os.path.join(PKG_MODELS, f"tabpfn-{model}.pt"))
    return TabPFNWideClassifier(device="cpu", save_attention_maps=mode, random_state=0, **source)


@pytest.fixture(scope="module", params=list(LABEL_CASES), ids=list(LABEL_CASES))
def full_and_label(request):
    case = LABEL_CASES[request.param]
    n = case["n_train"] + case["n_test"]
    X, y = make_classification(
        n_samples=n,
        n_features=case["n_features"],
        n_informative=5,
        n_classes=case["n_classes"],
        n_clusters_per_class=1,
        random_state=0,
    )
    X[np.random.default_rng(0).random(X.shape) < case["nan"]] = np.nan
    Xtr, Xte, ytr = X[: case["n_train"]], X[case["n_train"] :], y[: case["n_train"]]
    fitted = {}
    for mode in (True, "label"):
        clf = _classifier(case["model"], mode)
        clf.fit(Xtr, ytr)
        fitted[mode] = (clf, clf.predict_proba(Xte))
    return case, fitted[True], fitted["label"]


@pytest.mark.slow
def test_label_mode_predictions_match_full_mode(full_and_label):
    _, (_, proba_full), (_, proba_label) = full_and_label
    np.testing.assert_allclose(proba_label, proba_full, atol=1e-6)


@pytest.mark.slow
def test_full_map_label_row_equals_label_mode_mean_per_layer(full_and_label):
    case, (full, _), (label, _) = full_and_label
    n_rows = case["n_train"] + case["n_test"]
    raw_maps, _ = full.get_raw_attention_maps()
    assert len(raw_maps) == len(label.model.blocks)
    for raw_map, block in zip(raw_maps, label.model.blocks):
        rows_by_tokens = torch.cat(block.per_sample_attention_between_features.label_attention)
        assert rows_by_tokens.shape == (n_rows, raw_map.shape[0])
        np.testing.assert_allclose(
            rows_by_tokens.numpy().mean(axis=0),
            raw_map[-1] * case["n_train"] / n_rows,
            rtol=1e-4,
            atol=1e-8,
        )


@pytest.mark.slow
def test_get_attention_to_label_mean_identical_across_modes(full_and_label):
    case, (full, _), (label, _) = full_and_label
    n_rows = case["n_train"] + case["n_test"]
    from_full = full.get_attention_to_label(aggregation=np.mean) * case["n_train"] / n_rows
    direct = label.get_attention_to_label(aggregation=np.mean)
    assert direct.shape == (case["n_features"],)
    np.testing.assert_allclose(direct, from_full, rtol=1e-4, atol=1e-8)


@pytest.mark.slow
@pytest.mark.parametrize(
    "aggregation",
    [np.mean, np.median, functools.partial(np.quantile, q=0.9)],
    ids=["mean", "median", "q90"],
)
def test_label_mode_aggregation_matches_manual(full_and_label, aggregation):
    case, _, (label, _) = full_and_label
    mapping = label.get_feature_mapping_debug()["original_to_preprocessed"]
    per_layer = [
        aggregation(
            torch.cat(block.per_sample_attention_between_features.label_attention).numpy(), axis=0
        )[:-1]
        for block in label.model.blocks
    ]
    per_token = np.mean(per_layer, axis=0)
    expected = np.array([per_token[mapping[j]].mean() for j in range(case["n_features"])])
    np.testing.assert_allclose(
        label.get_attention_to_label(aggregation=aggregation), expected, rtol=1e-6
    )


@pytest.mark.slow
def test_median_differs_from_mean(full_and_label):
    _, _, (label, _) = full_and_label
    assert not np.allclose(
        label.get_attention_to_label(aggregation=np.median),
        label.get_attention_to_label(aggregation=np.mean),
    )


@pytest.mark.slow
def test_label_mode_reset_between_predicts(full_and_label):
    case, _, (label, _) = full_and_label
    first = label.get_attention_to_label()
    X, _ = make_classification(n_samples=7, n_features=case["n_features"], random_state=5)
    label.predict_proba(X)
    assert len(
        torch.cat(label.model.blocks[0].per_sample_attention_between_features.label_attention)
    ) == case["n_train"] + 7
    assert not np.allclose(label.get_attention_to_label(), first)


@pytest.mark.slow
def test_aggregation_guards(full_and_label):
    _, (full, _), (label, _) = full_and_label
    with pytest.raises(ValueError, match="require save_attention_maps='label'"):
        full.get_attention_to_label(aggregation=np.median)
    with pytest.raises(ValueError, match="require save_attention_maps=True"):
        label.get_attention_maps()
    with pytest.raises(AssertionError, match="must reduce"):
        label.get_attention_to_label(aggregation=lambda a, axis: a)


# ---------------------------------------------------------------------------
# Token-to-feature mapping. TabPFN's preprocessing drops constant columns,
# moves columns it detects as categorical (< 4 distinct values with > 100
# training rows) to the front when append_to_original is off, appends
# transformed copies when it is on, and adds SVD columns. The mapping must
# follow all of it.
# ---------------------------------------------------------------------------


def _wide_lowcard_data(n_samples, n_features, n_lowcard, n_const, seed=0):
    """Zero-dominated columns with values {0, 1, 2}, plus constant columns."""
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features, n_informative=10, random_state=seed
    )
    rng = np.random.default_rng(seed)
    special = rng.choice(n_features, n_lowcard + n_const, replace=False)
    lowcard, const = special[:n_lowcard], special[n_lowcard:]
    for j in lowcard:
        X[:, j] = rng.choice([0.0, 0.0, 0.0, 1.0, 2.0], size=n_samples)
    X[:, const] = 5.0
    return X, y, lowcard, const


MAPPING_CASES = {
    # 505 non-constant features >= 500: append_to_original resolves to False and
    # the categorical columns are moved to the front.
    "categoricals-moved": dict(n_features=520, append_to_original=False),
    "append-to-original": dict(n_features=200, append_to_original=True),
}


@pytest.mark.slow
@pytest.mark.parametrize("case", list(MAPPING_CASES))
def test_feature_mapping_follows_preprocessing(case):
    n_features = MAPPING_CASES[case]["n_features"]
    X, y, lowcard, const = _wide_lowcard_data(130, n_features, n_lowcard=150, n_const=15)
    Xtr, ytr = X[:120], y[:120]
    clf = _classifier("wide-v2-5k", False)
    clf.fit(Xtr, ytr)

    pipeline = clf.executor_.ensemble_members[0].cpu_preprocessor
    reshape = next(s for s, _ in pipeline.steps if hasattr(s, "append_to_original_decision_"))
    assert reshape.append_to_original_decision_ == MAPPING_CASES[case]["append_to_original"]
    final_columns = pipeline.final_feature_schema_.features
    assert sum(f.modality.value == "categorical" for f in final_columns) == len(lowcard)

    info = clf.get_feature_mapping_debug()
    assert info["n_preprocessed"] == pipeline.final_feature_schema_.num_columns
    for j in const:
        assert info["original_to_preprocessed"][j] == []

    # Every token assigned to a feature must be a deterministic one-to-one
    # function of that feature on the training rows.
    Xpre = pipeline.transform(Xtr).X
    n_checked = 0
    for j, positions in info["original_to_preprocessed"].items():
        if j not in const:
            assert positions, f"feature {j} has no token"
        for p in positions:
            n_pairs = len(set(zip(Xtr[:, j], Xpre[:, p])))
            assert n_pairs == len(np.unique(Xtr[:, j])) == len(np.unique(Xpre[:, p])), (
                f"token {p} ({info['token_names'][p]}) is not derived from feature {j}"
            )
            n_checked += 1
    assert n_checked == sum(f is not None for f in info["token_features"])


@pytest.mark.slow
def test_label_attention_survives_column_permutation():
    X, y, lowcard, const = _wide_lowcard_data(150, 200, n_lowcard=40, n_const=10, seed=1)
    perm = np.random.default_rng(2).permutation(X.shape[1])

    def scores(Xp):
        clf = TabPFNWideClassifier(
            model_name="v2",
            device="cpu",
            save_attention_maps="label",
            random_state=0,
            # Ordinal codes of detected categoricals depend on column order;
            # switch detection off to isolate the mapping.
            inference_config={"MIN_UNIQUE_FOR_NUMERICAL_FEATURES": 1},
        )
        clf.fit(Xp[:120], y[:120])
        clf.predict_proba(Xp[120:])
        return clf.get_attention_to_label()

    original = scores(X)
    permuted_back = np.empty_like(original)
    permuted_back[perm] = scores(X[:, perm])
    assert np.isnan(original[const]).all() and np.isnan(permuted_back[const]).all()
    keep = np.isfinite(original)
    assert spearmanr(original[keep], permuted_back[keep]).statistic > 0.9


@pytest.mark.slow
def test_full_maps_nan_for_dropped_features():
    X, y, _, const = _wide_lowcard_data(40, 30, n_lowcard=0, n_const=4, seed=3)
    clf = _classifier("v2", True)
    clf.fit(X[:30], y[:30])
    clf.predict_proba(X[30:])
    kept = np.setdiff1d(np.arange(30), const)
    for mapped in clf.get_attention_maps():
        assert mapped.shape == (30, 30)
        assert np.isnan(mapped[const]).all() and np.isnan(mapped[:, const]).all()
        assert np.isfinite(mapped[np.ix_(kept, kept)]).all()
    label = clf.get_attention_to_label()
    assert np.isnan(label[const]).all() and np.isfinite(label[kept]).all()
