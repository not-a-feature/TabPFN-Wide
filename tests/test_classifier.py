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
