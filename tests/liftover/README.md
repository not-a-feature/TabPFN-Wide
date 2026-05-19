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
