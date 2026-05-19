# TabPFN 6 → 8 liftover harness

This directory holds the side-by-side comparison harness used to verify that
the port from `tabpfn 6.0.6` / `tabpfnwide 0.1.0` to `tabpfn 8.x` /
current `tabpfnwide` produces equivalent predictions and attention maps.

It is **not** part of the pytest suite — it requires two isolated venvs with
incompatible `tabpfn` versions installed side by side and takes several
minutes to run. `pyproject.toml` excludes this directory from collection via
`norecursedirs`.

## Usage

```bash
# From the repo root:
./tests/liftover/compare_tabpfn_versions.sh                  # full run
./tests/liftover/compare_tabpfn_versions.sh --skip-install   # reuse venvs
./tests/liftover/compare_tabpfn_versions.sh --only-compare   # diff only
```

The venvs are created under `tests/liftover/.venvs/` and dumps land in
`tests/liftover/comparison_results/`. Both are gitignored.

## Files

- `compare_tabpfn_versions.sh` — driver. Creates venvs, installs the pinned
  package sets, runs `run_comparison.py` in each, then diffs.
- `run_comparison.py` — runs a fixed battery of `fit`/`predict` cases and
  dumps probabilities and attention maps to disk. Same script runs in both
  venvs (the public `tabpfnwide` API is stable across the two releases).
- `compare_results.py` — diffs two output trees with per-item tolerances and
  prints a verdict.

## Why raw attention maps differ
The raw maps are essentially identical — they're just stored in a different feature-token order.

Evidence (from tests/liftover/comparison_results/ for dummy_wide_wide_5k, layer 11):

Check	Cosine
Raw maps as-is (old vs new)	0.36
After one shared row+col permutation P	0.9998
A single permutation P recovered from layer 11 explains the divergence across every layer (cosines 0.99+ after applying P, layers 1–11). Only 2/125 positions in P are identity (~1.6%) — the orderings are completely different.

Root cause — the preprocessing pipeline shuffles features differently between tabpfn 6.0.6 and 8.0.3:

Run       	index_permutation_[:20]
old (6.0.6)	[101, 10, 18, 37, 99, 75, 60, 8, 35, 122, 1, 85, 70, 56, 65, 98, 32, 20, 72, 38]
new (8.0.3)	[101, 112, 23, 90, 34, 59, 6, 53, 74, 61, 3, 75, 45, 37, 55, 18, 122, 89, 9, 98]

Both use length 124 with append_to_original=True, both seeded random_state=0 — but TabPFN 8's preprocessing draws random numbers in a different order/with a different RNG path than 6.0.6, so the shuffle permutation is different. Raw attention maps are stored in post-shuffle token order, so they look very different even though they're encoding the same feature relationships.

That's why tabpfnwide/classifier.py:203 exists — get_attention_maps() applies P_norm @ raw @ P_norm.T to project back to original feature space, yielding cosine ~0.94 on the mapped maps. The remaining ~6% is small numerical drift (~1.6e-3 absolute max), likely from minor differences in preprocessing scaling/normalization between the two TabPFN versions.