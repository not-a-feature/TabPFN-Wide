#!/usr/bin/env bash
# Compare TabPFN-Wide outputs between tabpfn 6.0.6 (old, pip-installed
# tabpfnwide==0.1.0) and tabpfn 8.0.3 (new, local editable tabpfnwide).
#
# Creates two isolated virtual environments under tests/liftover/.venvs/,
# installs the pinned package set in each, runs the same battery of
# fits/predicts, and then diffs the dumped predictions and attention maps.
# Attention-map drift beyond the configured tolerance is reported as a failure.
#
# Usage (run from repo root):
#   ./tests/liftover/compare_tabpfn_versions.sh                  # full run
#   ./tests/liftover/compare_tabpfn_versions.sh --skip-install   # reuse existing venvs
#   ./tests/liftover/compare_tabpfn_versions.sh --only-compare   # skip both runs, just diff
#
# Designed to work in Git Bash on Windows (uses Scripts/python or bin/python).

set -euo pipefail

SKIP_INSTALL=0
ONLY_COMPARE=0
EXTRA_COMPARE_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-install) SKIP_INSTALL=1; shift ;;
        --only-compare) ONLY_COMPARE=1; shift ;;
        --strict-raw) EXTRA_COMPARE_ARGS+=("--strict-raw"); shift ;;
        --proba-tol) EXTRA_COMPARE_ARGS+=("--proba-tol" "$2"); shift 2 ;;
        --attn-tol) EXTRA_COMPARE_ARGS+=("--attn-tol" "$2"); shift 2 ;;
        -h|--help)
            sed -n '2,16p' "$0"
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

# Locate project root. Script lives at tests/liftover/, repo root is two levels up.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

VENV_DIR="$SCRIPT_DIR/.venvs"
RESULTS_DIR="$SCRIPT_DIR/comparison_results"
OLD_VENV="$VENV_DIR/tabpfn_old"
NEW_VENV="$VENV_DIR/tabpfn_new"
OLD_OUT="$RESULTS_DIR/old"
NEW_OUT="$RESULTS_DIR/new"

# Cross-platform venv-python locator: prefer Unix-style bin/, fall back to
# Windows Scripts/.
venv_python() {
    if [[ -x "$1/bin/python" ]]; then
        echo "$1/bin/python"
    elif [[ -x "$1/bin/python.exe" ]]; then
        echo "$1/bin/python.exe"
    elif [[ -x "$1/Scripts/python" ]]; then
        echo "$1/Scripts/python"
    elif [[ -x "$1/Scripts/python.exe" ]]; then
        echo "$1/Scripts/python.exe"
    else
        echo ""
    fi
}

create_venv() {
    local target="$1"
    if [[ -d "$target" ]]; then
        echo "venv already exists at $target"
        return 0
    fi
    echo ">>> Creating venv at $target"
    python -m venv "$target"
}

ensure_models_dir() {
    # tabpfnwide 0.1.0 expects checkpoints at ~/.tabpfnwide/models/, while the
    # editable install ships them in tabpfnwide/models/. Copy any local
    # checkpoints into the user cache so the old version finds them without
    # re-downloading.
    local src="$REPO_ROOT/tabpfnwide/models"
    local dst="$HOME/.tabpfnwide/models"
    if [[ -d "$src" ]]; then
        mkdir -p "$dst"
        for f in "$src"/tabpfn-wide-*.pt; do
            [[ -e "$f" ]] || continue
            local base
            base="$(basename "$f")"
            if [[ ! -e "$dst/$base" ]]; then
                echo "    copying $base -> $dst"
                cp "$f" "$dst/$base"
            fi
        done
    fi
}

run_old() {
    create_venv "$OLD_VENV"
    local py
    py="$(venv_python "$OLD_VENV")"
    if [[ -z "$py" ]]; then
        echo "Could not locate python in $OLD_VENV" >&2
        return 1
    fi
    if [[ $SKIP_INSTALL -eq 0 ]]; then
        echo ">>> Installing tabpfn==6.0.6 + tabpfnwide==0.1.0 into old venv"
        "$py" -m pip install --upgrade pip
        "$py" -m pip install "tabpfn==6.0.6" "tabpfnwide==0.1.0" \
            scikit-learn scipy numpy torch
    fi
    ensure_models_dir
    echo ">>> Running comparison harness in OLD venv"
    "$py" "$SCRIPT_DIR/run_comparison.py" --tag old --out "$OLD_OUT"
}

run_new() {
    create_venv "$NEW_VENV"
    local py
    py="$(venv_python "$NEW_VENV")"
    if [[ -z "$py" ]]; then
        echo "Could not locate python in $NEW_VENV" >&2
        return 1
    fi
    if [[ $SKIP_INSTALL -eq 0 ]]; then
        echo ">>> Installing local tabpfn (8.x) + local tabpfnwide into new venv"
        "$py" -m pip install --upgrade pip
        "$py" -m pip install -e "$REPO_ROOT/TabPFN"
        "$py" -m pip install -e "$REPO_ROOT" --no-deps
        # tabpfnwide pyproject.toml pins tabpfn==8.0.3 so --no-deps is required
        # when we've installed tabpfn editable from the local source.
        "$py" -m pip install scikit-learn scipy numpy torch
    fi
    ensure_models_dir
    echo ">>> Running comparison harness in NEW venv"
    "$py" "$SCRIPT_DIR/run_comparison.py" --tag new --out "$NEW_OUT"
}

run_compare() {
    # The compare script just needs numpy; use whichever venv we already have.
    local py
    py="$(venv_python "$NEW_VENV")"
    if [[ -z "$py" ]]; then
        py="$(venv_python "$OLD_VENV")"
    fi
    if [[ -z "$py" ]]; then
        py="python"
    fi
    echo ">>> Comparing $OLD_OUT vs $NEW_OUT"
    "$py" "$SCRIPT_DIR/compare_results.py" \
        --old "$OLD_OUT" --new "$NEW_OUT" \
        --report "$RESULTS_DIR/diff_report.json" \
        "${EXTRA_COMPARE_ARGS[@]}"
}

mkdir -p "$RESULTS_DIR"

if [[ $ONLY_COMPARE -eq 0 ]]; then
    run_old
    run_new
fi

run_compare
