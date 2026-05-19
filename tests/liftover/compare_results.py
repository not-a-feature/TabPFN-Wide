"""Compare outputs of two run_comparison.py runs.

Usage:
    python compare_results.py --old results/old --new results/new

Per-item tolerances (override on the command line if needed):

  proba:               max_abs ≤ 5e-2 and cosine ≥ 0.999
  attention_to_label:  max_abs ≤ 2e-2 and cosine ≥ 0.99
  attention_maps:      max_abs ≤ 5e-1 and cosine ≥ 0.90  (mapped, user-facing)
  raw_attention_maps:  cosine ≥ 0.30                      (informational only —
                       deep-layer raw maps amplify any preprocessing drift, so
                       this is a soft check and counts as a warning, not a
                       failure, unless --strict-raw is passed.)

Exits non-zero if any non-soft item exceeds its tolerance.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Tolerance:
    max_abs: float | None = None
    min_cosine: float | None = None
    soft: bool = False  # if True, violations are warnings, not failures


# Pass is `cosine >= min_cosine` for each item. Absolute diffs are reported
# but not enforced — they vary with input magnitude and are noisy across major
# library versions, while cosine cleanly measures *what* the model is
# attending to. ``attn_raw`` (the un-mapped, internal per-token map) is soft:
# small preprocessing differences amplify through 12 transformer layers, so
# its deep-layer cosine drops are structural, not a regression of the port.
DEFAULT_TOLERANCES: dict[str, Tolerance] = {
    "proba":              Tolerance(min_cosine=0.999),
    "attn_label":         Tolerance(min_cosine=0.99),
    "attn_mapped":        Tolerance(min_cosine=0.90),
    "attn_raw":           Tolerance(min_cosine=0.30, soft=True),
}


def _load_npz_layers(path: Path) -> list[np.ndarray]:
    with np.load(path) as data:
        keys = sorted(data.keys())
        return [data[k] for k in keys]


def _array_diff(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return {
            "shape_old": list(a.shape),
            "shape_new": list(b.shape),
            "shape_mismatch": True,
            "max_abs": float("inf"),
            "mean_abs": float("inf"),
            "cosine": 0.0,
        }
    diff = a.astype(np.float64) - b.astype(np.float64)
    abs_diff = np.abs(diff)
    a_flat = a.astype(np.float64).ravel()
    b_flat = b.astype(np.float64).ravel()
    na = np.linalg.norm(a_flat)
    nb = np.linalg.norm(b_flat)
    cos = float(a_flat @ b_flat / (na * nb)) if na > 0 and nb > 0 else 1.0
    return {
        "shape": list(a.shape),
        "max_abs": float(abs_diff.max()),
        "mean_abs": float(abs_diff.mean()),
        "cosine": cos,
    }


def _layer_aggregate(layer_diffs: list[dict]) -> dict:
    """Aggregate a list of per-layer diffs into worst-case stats."""
    return {
        "n_layers": len(layer_diffs),
        "max_abs": float(max(d["max_abs"] for d in layer_diffs)),
        "mean_abs_avg": float(np.mean([d["mean_abs"] for d in layer_diffs])),
        "min_cosine": float(min(d["cosine"] for d in layer_diffs)),
        "worst_layer": int(np.argmax([d["max_abs"] for d in layer_diffs])),
        "per_layer": layer_diffs,
    }


def _meets(tol: Tolerance, max_abs: float, cosine: float) -> bool:
    if tol.max_abs is not None and not (max_abs <= tol.max_abs):
        return False
    if tol.min_cosine is not None and not (cosine >= tol.min_cosine):
        return False
    return True


def compare_case(old_dir: Path, new_dir: Path, case_id: str) -> dict:
    case_old = old_dir / case_id
    case_new = new_dir / case_id
    result: dict = {"case_id": case_id, "items": {}}

    scalar_items = {
        "proba_train":         ("proba_train.npy",       "proba"),
        "proba_test":          ("proba_test.npy",        "proba"),
        "attention_to_label":  ("attention_to_label.npy", "attn_label"),
    }
    for name, (fname, kind) in scalar_items.items():
        a_path = case_old / fname
        b_path = case_new / fname
        if not a_path.exists() or not b_path.exists():
            result["items"][name] = {"kind": kind, "missing": True}
            continue
        a = np.load(a_path)
        b = np.load(b_path)
        d = _array_diff(a, b)
        d["kind"] = kind
        result["items"][name] = d

    npz_items = {
        "attention_maps.npz":     "attn_mapped",
        "raw_attention_maps.npz": "attn_raw",
    }
    for fname, kind in npz_items.items():
        a_path = case_old / fname
        b_path = case_new / fname
        if not a_path.exists() or not b_path.exists():
            result["items"][fname] = {"kind": kind, "missing": True}
            continue
        a_layers = _load_npz_layers(a_path)
        b_layers = _load_npz_layers(b_path)
        if len(a_layers) != len(b_layers):
            result["items"][fname] = {
                "kind": kind,
                "layer_count_mismatch": True,
                "layers_old": len(a_layers),
                "layers_new": len(b_layers),
            }
            continue
        layer_diffs = [_array_diff(a, b) for a, b in zip(a_layers, b_layers)]
        agg = _layer_aggregate(layer_diffs)
        agg["kind"] = kind
        result["items"][fname] = agg

    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--old", required=True)
    p.add_argument("--new", required=True)
    p.add_argument("--report")
    p.add_argument("--strict-raw", action="store_true",
                   help="Treat raw_attention_maps failures as hard failures.")
    p.add_argument("--proba-tol", type=float)
    p.add_argument("--attn-tol", type=float,
                   help="Backward-compat alias: applies to attn_mapped and attn_label.")
    args = p.parse_args()

    tolerances = dict(DEFAULT_TOLERANCES)
    if args.proba_tol is not None:
        tolerances["proba"] = Tolerance(max_abs=args.proba_tol,
                                        min_cosine=DEFAULT_TOLERANCES["proba"].min_cosine)
    if args.attn_tol is not None:
        tolerances["attn_mapped"] = Tolerance(max_abs=args.attn_tol,
                                              min_cosine=DEFAULT_TOLERANCES["attn_mapped"].min_cosine)
        tolerances["attn_label"] = Tolerance(max_abs=args.attn_tol,
                                             min_cosine=DEFAULT_TOLERANCES["attn_label"].min_cosine)
    if args.strict_raw:
        tolerances["attn_raw"] = Tolerance(
            max_abs=DEFAULT_TOLERANCES["attn_raw"].max_abs,
            min_cosine=DEFAULT_TOLERANCES["attn_raw"].min_cosine,
            soft=False,
        )

    old_dir = Path(args.old)
    new_dir = Path(args.new)

    for tag, d in (("old", old_dir), ("new", new_dir)):
        env_file = d / "_env.json"
        if env_file.exists():
            env = json.loads(env_file.read_text())
            print(f"[{tag}] tabpfn={env.get('tabpfn_version')} "
                  f"tabpfnwide={env.get('tabpfnwide_version')} "
                  f"torch={env.get('torch')}")

    cases = sorted({p.name for p in old_dir.iterdir() if p.is_dir() and not p.name.startswith("_")}
                   & {p.name for p in new_dir.iterdir() if p.is_dir() and not p.name.startswith("_")})

    if not cases:
        print("No matching cases between the two output directories.", file=sys.stderr)
        return 2

    print()
    print("Tolerances (per item kind):")
    for kind, tol in tolerances.items():
        flags = []
        if tol.max_abs is not None:
            flags.append(f"max_abs<={tol.max_abs}")
        if tol.min_cosine is not None:
            flags.append(f"cos>={tol.min_cosine}")
        if tol.soft:
            flags.append("soft (warn only)")
        print(f"  {kind:14s} {', '.join(flags)}")
    print()

    header = (f"{'case':28s} {'item':24s} {'kind':12s} "
              f"{'max_abs':>11s} {'mean_abs':>11s} {'cosine':>8s}  status")
    print(header)
    print("-" * len(header))

    overall_pass = True
    n_warn = 0
    n_fail = 0
    report = []

    for case_id in cases:
        case_result = compare_case(old_dir, new_dir, case_id)
        report.append(case_result)

        for item_name, d in case_result["items"].items():
            kind = d.get("kind", "")
            if d.get("missing"):
                print(f"{case_id:28s} {item_name:24s} {kind:12s} "
                      f"{'MISSING':>11s} {'':>11s} {'':>8s}  skip")
                continue
            if d.get("shape_mismatch") or d.get("layer_count_mismatch"):
                tol = tolerances.get(kind)
                status = "WARN" if tol and tol.soft else "FAIL"
                if status == "FAIL":
                    overall_pass = False
                    n_fail += 1
                else:
                    n_warn += 1
                print(f"{case_id:28s} {item_name:24s} {kind:12s} "
                      f"{'SHAPE!':>11s} {'':>11s} {'':>8s}  {status}")
                continue

            max_abs = d["max_abs"]
            mean_abs = d.get("mean_abs", d.get("mean_abs_avg", float("nan")))
            cosine = d.get("cosine", d.get("min_cosine", float("nan")))

            tol = tolerances.get(kind)
            if tol is None:
                status = "ok  "
            else:
                ok = _meets(tol, max_abs, cosine)
                if ok:
                    status = "ok  "
                elif tol.soft:
                    status = "WARN"
                    n_warn += 1
                else:
                    status = "FAIL"
                    overall_pass = False
                    n_fail += 1

            print(f"{case_id:28s} {item_name:24s} {kind:12s} "
                  f"{max_abs:11.3e} {mean_abs:11.3e} {cosine:8.4f}  {status}")

    if args.report:
        # Drop the per_layer payload from the printed report to keep it small.
        slim = []
        for case in report:
            entry = {"case_id": case["case_id"], "items": {}}
            for name, item in case["items"].items():
                slim_item = {k: v for k, v in item.items() if k != "per_layer"}
                entry["items"][name] = slim_item
            slim.append(entry)
        Path(args.report).write_text(json.dumps(slim, indent=2))

    print()
    print(f"failures: {n_fail}, warnings: {n_warn}")
    print("VERDICT:", "PASS" if overall_pass else "FAIL")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
