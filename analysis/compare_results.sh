#!/bin/bash

# Script to run comparison plotting across all analysis results

# Default results directory
RESULTS_DIR="${1:-analysis_results}"

echo "Running comparison plotting on directory: $RESULTS_DIR"

python analysis/plot_results.py --input_dir "$RESULTS_DIR" --compare_mode

echo "Comparison plotting completed. Check ${RESULTS_DIR}/comparison_plots for output."
