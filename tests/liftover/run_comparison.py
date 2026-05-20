"""Run a fixed battery of TabPFN-Wide tests and dump predictions + attention
maps to disk.

Designed to be invoked twice — once in a venv with tabpfn 6.0.6 / tabpfnwide
0.1.0 and once with tabpfn 8.0.3 / current tabpfnwide — so the outputs can be
diffed with compare_results.py.

The public tabpfnwide API used here (TabPFNWideClassifier, fit, predict_proba,
get_attention_maps, get_attention_to_label) is identical between the two
releases, so this script runs as-is in either environment.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _save_attention_maps(path: Path, maps) -> None:
    """Save a list of 2D arrays as an npz with keys layer_000, layer_001, ..."""
    arr_dict = {f"layer_{i:03d}": np.asarray(m) for i, m in enumerate(maps)}
    np.savez_compressed(path, **arr_dict)


def _dataset_dummy_small():
    X, y = make_classification(
        n_samples=40,
        n_features=8,
        n_informative=4,
        random_state=0,
        shuffle=False,
    )
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    return Xtr, Xte, ytr, yte


def _dataset_dummy_wide():
    """Wide dataset: few samples, many features."""
    X, y = make_classification(
        n_samples=30,
        n_features=60,
        n_informative=5,
        random_state=0,
        shuffle=False,
    )
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    return Xtr, Xte, ytr, yte


def _dataset_breast_cancer():
    """20 train / 30 test split from sklearn's breast cancer dataset."""
    data = load_breast_cancer()
    Xtr, Xte, ytr, yte = train_test_split(
        data.data, data.target, test_size=0.85, random_state=0, stratify=data.target
    )
    # Keep test set small for speed
    return Xtr, Xte[:30], ytr, yte[:30]


# Case format: (case_id, dataset_fn, model_name, save_attention)
CASES = [
    ("dummy_small_v2", _dataset_dummy_small, "v2", True),
    ("dummy_small_wide_1_5k", _dataset_dummy_small, "wide-v2-1.5k", True),
    ("dummy_wide_wide_5k", _dataset_dummy_wide, "wide-v2-5k", True),
    ("breast_cancer_wide_5k", _dataset_breast_cancer, "wide-v2-5k", True),
]


def run_case(
    case_id: str, dataset_fn, model_name: str, save_attention: bool, out_dir: Path
) -> dict:
    """Fit on (Xtr, ytr), predict on Xtr and Xte, dump everything, return
    timing/metadata."""
    from tabpfnwide.classifier import TabPFNWideClassifier  # imported lazily

    case_dir = out_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(0)
    Xtr, Xte, ytr, yte = dataset_fn()

    # Save inputs once per tag — useful as a sanity check.
    np.save(case_dir / "X_train.npy", Xtr)
    np.save(case_dir / "X_test.npy", Xte)
    np.save(case_dir / "y_train.npy", ytr)
    np.save(case_dir / "y_test.npy", yte)

    kwargs = dict(
        model_name=model_name,
        device="cpu",
        n_estimators=1,
        features_per_group=1,
        save_attention_maps=save_attention,
        random_state=0,
    )

    print(f"  [{case_id}] model={model_name} save_attn={save_attention}")
    t0 = time.perf_counter()
    clf = TabPFNWideClassifier(**kwargs)
    t1 = time.perf_counter()
    clf.fit(Xtr, ytr)
    t2 = time.perf_counter()
    proba_test = clf.predict_proba(Xte)
    t3 = time.perf_counter()

    np.save(case_dir / "proba_test.npy", proba_test)

    info = {
        "case_id": case_id,
        "model_name": model_name,
        "n_train": int(Xtr.shape[0]),
        "n_test": int(Xte.shape[0]),
        "n_features": int(Xtr.shape[1]),
        "init_seconds": t1 - t0,
        "fit_seconds": t2 - t1,
        "predict_seconds": t3 - t2,
        "proba_test_shape": list(proba_test.shape),
    }

    if save_attention:
        maps = clf.get_attention_maps()
        if maps is not None:
            _save_attention_maps(case_dir / "attention_maps.npz", maps)
            info["attention_maps_layers"] = len(maps)
            info["attention_maps_shape"] = list(np.asarray(maps[0]).shape)

        try:
            raw_maps, _ = clf.get_raw_attention_maps()
            if raw_maps is not None:
                _save_attention_maps(case_dir / "raw_attention_maps.npz", raw_maps)
                info["raw_attention_maps_layers"] = len(raw_maps)
        except Exception as e:  # noqa: BLE001
            info["raw_attention_maps_error"] = repr(e)

        try:
            attn_to_label = clf.get_attention_to_label()
            np.save(case_dir / "attention_to_label.npy", attn_to_label)
            info["attention_to_label_shape"] = list(attn_to_label.shape)
        except Exception as e:  # noqa: BLE001
            info["attention_to_label_error"] = repr(e)

    # Now safe to do the second predict — attention reads are done.
    proba_train = clf.predict_proba(Xtr)
    np.save(case_dir / "proba_train.npy", proba_train)
    info["proba_train_shape"] = list(proba_train.shape)

    with (case_dir / "info.json").open("w") as f:
        json.dump(info, f, indent=2)

    return info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="Label for this run (e.g. 'old', 'new').")
    parser.add_argument("--out", required=True, help="Output directory.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import tabpfn, tabpfnwide  # noqa: PLC0415

    env_info = {
        "tag": args.tag,
        "tabpfn_version": getattr(tabpfn, "__version__", "unknown"),
        "tabpfnwide_version": getattr(tabpfnwide, "__version__", "unknown"),
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    print(
        f"[{args.tag}] tabpfn={env_info['tabpfn_version']} "
        f"tabpfnwide={env_info['tabpfnwide_version']}"
    )
    with (out_dir / "_env.json").open("w") as f:
        json.dump(env_info, f, indent=2)

    failures = []
    for case_id, dataset_fn, model_name, save_attention in CASES:
        try:
            run_case(case_id, dataset_fn, model_name, save_attention, out_dir)
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            failures.append((case_id, repr(e)))

    if failures:
        print(f"\n[{args.tag}] {len(failures)} case(s) failed:")
        for cid, err in failures:
            print(f"  - {cid}: {err}")
        return 1

    print(f"\n[{args.tag}] all {len(CASES)} cases completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
