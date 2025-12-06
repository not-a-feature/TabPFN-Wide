#!/bin/bash
#
# Pipeline to run all TabPFN-Wide analysis scripts
# This script orchestrates the execution of all 6 analysis scripts with configurable parameters.
#
# Usage: ./run_all_analysis.sh
#
# Before running, configure the parameters below to match your setup.

set -e  # Exit on error

# ============================================================================
# CONFIGURATION SECTION - Modify these parameters as needed
# ============================================================================

# Directory paths
CHECKPOINT_DIR="checkpoints"
OUTPUT_DIR="analysis_results"
BENCHMARK_DATA_DIR="benchmark_data"

# Device configuration
DEVICE="cuda:0"

# HDLSS Benchmark Configuration
HDLSS_DATA_DIR="${BENCHMARK_DATA_DIR}/hdlss_new_data"
HDLSS_OUTPUT_FILE="${OUTPUT_DIR}/hdlss_benchmark_results.csv"
HDLSS_MAX_FEATURES=50000
HDLSS_MIN_FEATURES=0
HDLSS_MAX_INSTANCES=10000

# Multi-omics Feature Reduction Configuration
MULTIOMICS_DATASET_NAME="BRCA"  # Options: BRCA, COAD, GBM, LGG, OV
MULTIOMICS_OUTPUT_FILE="${OUTPUT_DIR}/multiomics_feature_reduction_results.csv"
MULTIOMICS_OMICS_LIST="mRNA cnv"  # Space-separated list of omics types

# OpenML Benchmark Configuration
OPENML_SUITE_ID=337  # OpenML suite ID
OPENML_OUTPUT_FILE="${OUTPUT_DIR}/openml_benchmark_results.csv"
OPENML_MAX_FEATURES=500
OPENML_MIN_FEATURES=0
OPENML_MAX_INSTANCES=10000

# OpenML Widening Configuration
OPENML_WIDENING_DATASET_IDS="1494 40536"  # Space-separated list of dataset IDs
OPENML_WIDENING_OUTPUT_FOLDER="${OUTPUT_DIR}/openml_widening"
OPENML_WIDENING_SPARSITY=0.01
OPENML_WIDENING_FEATURE_NUMBERS="0 50 500 2000 5000 10000 20000 30000"

# Multi-omics Attention Extraction Configuration
ATTENTION_MULTIOMICS_DATASET="BRCA"
ATTENTION_MULTIOMICS_OUTPUT="${OUTPUT_DIR}/multiomics_attention.pt"
ATTENTION_MULTIOMICS_CHECKPOINT="${CHECKPOINT_DIR}/best_model.pt"  # Specify your checkpoint
ATTENTION_MULTIOMICS_OMIC="mrna"

# Widening Attention Extraction Configuration
ATTENTION_WIDENING_OPENML_ID=1494
ATTENTION_WIDENING_CHECKPOINT="${CHECKPOINT_DIR}/best_model.pt"  # Specify your checkpoint
ATTENTION_WIDENING_OUTPUT="${OUTPUT_DIR}/widening_attention.pkl"

# ============================================================================
# SETUP
# ============================================================================

echo "========================================"
echo "TabPFN-Wide Analysis Pipeline"
echo "========================================"
echo ""

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OPENML_WIDENING_OUTPUT_FOLDER}"

# Get the base directory (where this script is located)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

# ============================================================================
# SCRIPT 1: HDLSS Benchmark
# ============================================================================

echo "----------------------------------------"
echo "1. Running HDLSS Benchmark..."
echo "----------------------------------------"
python analysis/hdlss_benchmark.py \
    "${HDLSS_DATA_DIR}" \
    "${HDLSS_OUTPUT_FILE}" \
    --max_features ${HDLSS_MAX_FEATURES} \
    --min_features ${HDLSS_MIN_FEATURES} \
    --max_instances ${HDLSS_MAX_INSTANCES} \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --device "${DEVICE}"

echo "✓ HDLSS Benchmark completed. Results saved to ${HDLSS_OUTPUT_FILE}"
echo ""

# ============================================================================
# SCRIPT 2: Multi-omics Feature Reduction
# ============================================================================

echo "----------------------------------------"
echo "2. Running Multi-omics Feature Reduction..."
echo "----------------------------------------"
python analysis/multiomics_feature_reduction.py \
    "${MULTIOMICS_DATASET_NAME}" \
    "${CHECKPOINT_DIR}" \
    "${MULTIOMICS_OUTPUT_FILE}" \
    ${MULTIOMICS_OMICS_LIST} \
    --device "${DEVICE}"

echo "✓ Multi-omics Feature Reduction completed. Results saved to ${MULTIOMICS_OUTPUT_FILE}"
echo ""

# ============================================================================
# SCRIPT 3: OpenML Benchmark
# ============================================================================

echo "----------------------------------------"
echo "3. Running OpenML Benchmark..."
echo "----------------------------------------"
python analysis/openml_benchmark.py \
    ${OPENML_SUITE_ID} \
    ${OPENML_MAX_FEATURES} \
    ${OPENML_MIN_FEATURES} \
    ${OPENML_MAX_INSTANCES} \
    "${OPENML_OUTPUT_FILE}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --device "${DEVICE}"

echo "✓ OpenML Benchmark completed. Results saved to ${OPENML_OUTPUT_FILE}"
echo ""

# ============================================================================
# SCRIPT 4: OpenML Widening
# ============================================================================

echo "----------------------------------------"
echo "4. Running OpenML Widening..."
echo "----------------------------------------"
python analysis/openml_widening.py \
    ${OPENML_WIDENING_DATASET_IDS} \
    --output_folder "${OPENML_WIDENING_OUTPUT_FOLDER}" \
    --device "${DEVICE}" \
    --checkpoints_dir "${CHECKPOINT_DIR}" \
    --sparsity ${OPENML_WIDENING_SPARSITY} \
    --feature_numbers ${OPENML_WIDENING_FEATURE_NUMBERS}

echo "✓ OpenML Widening completed. Results saved to ${OPENML_WIDENING_OUTPUT_FOLDER}"
echo ""

# ============================================================================
# SCRIPT 5: Extract Multi-omics Attention
# ============================================================================

echo "----------------------------------------"
echo "5. Extracting Multi-omics Attention..."
echo "----------------------------------------"
python analysis/extract_multi_omics_attention.py \
    --dataset_name "${ATTENTION_MULTIOMICS_DATASET}" \
    --output_file "${ATTENTION_MULTIOMICS_OUTPUT}" \
    --checkpoint_path "${ATTENTION_MULTIOMICS_CHECKPOINT}" \
    --device "${DEVICE}" \
    --omic "${ATTENTION_MULTIOMICS_OMIC}"

echo "✓ Multi-omics Attention extraction completed. Results saved to ${ATTENTION_MULTIOMICS_OUTPUT}"
echo ""

# ============================================================================
# SCRIPT 6: Extract Widening Attention
# ============================================================================

echo "----------------------------------------"
echo "6. Extracting Widening Attention..."
echo "----------------------------------------"
python analysis/extract_widening_attention.py \
    --device "${DEVICE}" \
    --openml_id ${ATTENTION_WIDENING_OPENML_ID} \
    --checkpoint_path "${ATTENTION_WIDENING_CHECKPOINT}" \
    --output "${ATTENTION_WIDENING_OUTPUT}"

echo "✓ Widening Attention extraction completed. Results saved to ${ATTENTION_WIDENING_OUTPUT}"
echo ""

# ============================================================================
# COMPLETION
# ============================================================================

echo "========================================"
echo "All analysis scripts completed successfully!"
echo "========================================"
echo ""
echo "Results summary:"
echo "  - HDLSS Benchmark: ${HDLSS_OUTPUT_FILE}"
echo "  - Multi-omics Feature Reduction: ${MULTIOMICS_OUTPUT_FILE}"
echo "  - OpenML Benchmark: ${OPENML_OUTPUT_FILE}"
echo "  - OpenML Widening: ${OPENML_WIDENING_OUTPUT_FOLDER}"
echo "  - Multi-omics Attention: ${ATTENTION_MULTIOMICS_OUTPUT}"
echo "  - Widening Attention: ${ATTENTION_WIDENING_OUTPUT}"
echo ""
