#!/bin/bash

#SBATCH --job-name=TabPFN-Wide-analysis
#SBATCH --cpus-per-task=100
#SBATCH --nodes=1
#SBATCH --partition=h100-ferranti # --partition=h100-ferranti h100-preemptable-ferranti
#SBATCH --mem=100G

#SBATCH --exclude=mlcbm[004-005] # Broken Node

#SBATCH --gres=gpu:1

#SBATCH --time=24:00:00

#SBATCH --error=job.%J.err 
#SBATCH --output=job.%J.out 

#SBATCH --mail-type=ALL 
#SBATCH --mail-user=jules.kreuer@uni-tuebingen.de 

# Source init script
source training/init.sh

# Define Jobs: Type:Value
# TODO Uppdate array size
JOBS=(
    "MODEL:realmlp"
)
SLURM_ARRAY_TASK_ID=0
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#JOBS[@]}" ]; then
    echo "Invalid array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

JOB_INFO=${JOBS[$SLURM_ARRAY_TASK_ID]}
JOB_TYPE=${JOB_INFO%%:*}
JOB_VALUE=${JOB_INFO#*:}

if [ "$JOB_TYPE" == "MODEL" ]; then
    CHECKPOINT_PATH="$JOB_VALUE"
    OUTPUT_DIR="${BASE_DIR_LOCAL}/analysis_results/${JOB_VALUE}"
else
    echo "Unknown job type: $JOB_TYPE"
    exit 1
fi

echo "Running analysis for: $JOB_INFO"
echo "Checkpoint/Model: $CHECKPOINT_PATH"
echo "Output directory: $OUTPUT_DIR"

bash analysis/run_analysis.sh "$CHECKPOINT_PATH" "$OUTPUT_DIR"
