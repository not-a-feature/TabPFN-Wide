#!/bin/bash

set -e

export PYTHONPATH=$PYTHONPATH:.

CHECKPOINT_PATH="$1"
OUTPUT_DIR="$2"

if [ -z "$CHECKPOINT_PATH" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 <checkpoint_path> <output_dir>"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"

# HDLSS Benchmark

if [ ! -f "${OUTPUT_DIR}/hdlss_benchmark_results.csv" ]; then
    echo "----------------------------------------"
    echo "Running HDLSS Benchmark..."
    echo "----------------------------------------"
    python analysis/hdlss_benchmark.py \
        "benchmark_data/hdlss_new_data" \
        "${OUTPUT_DIR}/hdlss_benchmark_results.csv" \
        --checkpoint_path "$CHECKPOINT_PATH" 
else
    echo "HDLSS Benchmark results exist. Skipping."
fi


# Multi-omics Feature Reduction

if [ ! -f "${OUTPUT_DIR}/multiomics_feature_reduction_results.csv" ]; then
    echo "----------------------------------------"
    echo "Running Multi-omics Feature Reduction..."
    echo "----------------------------------------"
    for dataset in BRCA COAD GBM LGG OV; do
        echo "Processing dataset: $dataset"
        python analysis/multiomics_feature_reduction.py \
            "benchmark_data/multiomics_benchmark_data" \
            "${OUTPUT_DIR}/multiomics_feature_reduction_results.csv" \
            --checkpoint_path "$CHECKPOINT_PATH" \
            --dataset "$dataset" \
            --omics "mrna"
    done
else
    echo "Multi-omics Feature Reduction results exist. Skipping."
fi


# OpenML Benchmark
if [ ! -f "${OUTPUT_DIR}/openml_benchmark_results.csv" ]; then
    echo "----------------------------------------"
    echo "Running OpenML Benchmark..."
    echo "----------------------------------------"
    python analysis/openml_benchmark.py \
        "${OUTPUT_DIR}/openml_benchmark_results.csv" \
        --suite_id 457 \
        --checkpoint_path "$CHECKPOINT_PATH"
else
    echo "OpenML Benchmark results exist. Skipping."
fi

# OpenML Widening with multiple sparsity values
SPARSITIES=(0 0.02 0.25 0.5)
for SPARSITY in "${SPARSITIES[@]}"; do
    WIDENING_DIR="${OUTPUT_DIR}/openml_widening/sparsity_${SPARSITY}"
    if [ ! -d "$WIDENING_DIR" ]; then
        echo "----------------------------------------"
        echo "Running OpenML Widening with sparsity=${SPARSITY}..."
        echo "----------------------------------------"
        python analysis/openml_widening.py \
            "$WIDENING_DIR" \
            --dataset_ids 54 188 1049 1067 1468 1494 40982 40984 41157 46921 46930 46940 46980 \
            --checkpoint_path "$CHECKPOINT_PATH" \
            --sparsity "$SPARSITY"
    else
        echo "OpenML Widening (sparsity=${SPARSITY}) results exist. Skipping."
    fi
done

#Multi-omics Attention Extraction
# if [ ! -f "${OUTPUT_DIR}/multiomics_attention.pt" ]; then
#     echo "----------------------------------------"
#     echo "Running Multi-omics Attention Extraction..."
#     echo "----------------------------------------"
#     for dataset in BRCA COAD GBM LGG OV; do
#         echo "Processing dataset: $dataset"
#         python analysis/extract_multi_omics_attention.py \
#             "benchmark_data/multiomics_benchmark_data" \
#             "${OUTPUT_DIR}/multiomics_attention" \
#             --checkpoint_path "$CHECKPOINT_PATH" \
#             --dataset "$dataset" \
#             --omic "mrna"
#     done
# else
#     echo "Multi-omics Attention Extraction results exist. Skipping."
# fi

# Widening Attention Extraction
if [ "$CHECKPOINT_PATH" != "tabicl" ] && [ "$CHECKPOINT_PATH" != "random_forest" ] && [ "$CHECKPOINT_PATH" != "xgboost" ] && [ "$CHECKPOINT_PATH" != "realmlp" ]; then
if [ ! -f "${OUTPUT_DIR}/widening_attention.pkl" ]; then
    echo "----------------------------------------"
    echo "Running Widening Attention Extraction..."
    echo "----------------------------------------"
    python analysis/extract_widening_attention.py \
        "${OUTPUT_DIR}/widening_attention.pkl" \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --openml_id 1494
else
    echo "Widening Attention Extraction results exist. Skipping."
fi
fi

# SNP Benchmark

if [ ! -f "${OUTPUT_DIR}/snp_benchmark_results.csv" ]; then
    echo "----------------------------------------"
    echo "Running SNP Benchmark..."
    echo "----------------------------------------"
    python analysis/snp_benchmark.py \
        "benchmark_data/snp_data" \
        "${OUTPUT_DIR}/snp_benchmark_results.csv" \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --device "cuda:0"
else
    echo "SNP Benchmark results exist. Skipping."
fi


# Plotting Results
echo "----------------------------------------"
echo "Plotting Results..."
echo "----------------------------------------"
python analysis/plot_results.py --input_dir "$OUTPUT_DIR"

echo "All analysis completed."
