"""Pytest tests for TabPFNWideClassifier argument validation and basic behaviour.

The validation tests are cheap (no model load). The end-to-end v2 test downloads
the base TabPFN-v2 weights and is marked ``slow`` so it can be skipped with
``pytest -m "not slow"``.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from sklearn.datasets import make_classification

from tabpfnwide.classifier import VALID_MODELS, TabPFNWideClassifier


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

    # Sanity: patch is installed on each attention module that has the wide
    # between-features attention, with save_att_map=True.
    patched_modules = [
        layer.self_attn_between_features
        for layer in clf._wide_model.transformer_encoder.layers
        if hasattr(layer, "self_attn_between_features")
    ]
    assert patched_modules, "no between-features attention modules found"
    assert all(getattr(m, "save_att_map", False) for m in patched_modules)

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
