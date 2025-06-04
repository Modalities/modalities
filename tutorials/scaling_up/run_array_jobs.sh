#!/bin/sh
#
# submit_experiments.sh
#
# Submits separate Slurm jobs for each desired node count.

set -euo pipefail

# Initialize variables
CONFIG_FILE_PATH=""
EXPERIMENT_FOLDER_PATH=""
PYTHON_ENV_PATH=""
NUM_WARMUP_STEPS=""
NUM_MEASUREMENT_STEPS=""
NODE_COUNTS=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config_file)
            CONFIG_FILE_PATH="$2"
            shift 2
            ;;
        --experiment_folder)
            EXPERIMENT_FOLDER_PATH="$2"
            shift 2
            ;;
        --python_env_path)
            PYTHON_ENV_PATH="$2"
            shift 2
            ;;
        --num_warmup_steps)
            NUM_WARMUP_STEPS="$2"
            shift 2
            ;;
        --num_measurement_steps)
            NUM_MEASUREMENT_STEPS="$2"
            shift 2
            ;;
        --node_counts)
            NODE_COUNTS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Enforce required arguments
if [[ -z "$CONFIG_FILE_PATH" ]]; then
    echo "Error: --config_file is required."
    exit 1
fi

if [[ -z "$EXPERIMENT_FOLDER_PATH" ]]; then
    echo "Error: --experiment_folder is required."
    exit 1
fi

if [[ -z "$PYTHON_ENV_PATH" ]]; then
    echo "Error: --python_env_path is required."
    exit 1
fi

if [[ -z "$NUM_WARMUP_STEPS" ]]; then
    echo "Error: --num_warmup_steps is required."
    exit 1
fi

if [[ -z "$NUM_MEASUREMENT_STEPS" ]]; then
    echo "Error: --num_measurement_steps is required."
    exit 1
fi

if [[ -z "$NODE_COUNTS" ]]; then
    echo "Error: --node_counts is required (e.g., '1 2 4')."
    exit 1
fi

DATE_OF_RUN=$(date +"%Y-%m-%d__%H-%M-%S")

# Calculate short hash of the config file
CONFIG_HASH=$(sha256sum "$CONFIG_FILE_PATH" | awk '{print substr($1, 1, 8)}')

# Append to EXPERIMENT_FOLDER_PATH
EXPERIMENT_FOLDER_PATH="${EXPERIMENT_FOLDER_PATH}/${DATE_OF_RUN}_${CONFIG_HASH}"

echo "Using experiment folder path: $EXPERIMENT_FOLDER_PATH"

# Submit jobs
for NUM_NODES in $NODE_COUNTS; do
    echo "Submitting job with $NUM_NODES node(s)..."
    sbatch --nodes=$NUM_NODES \
           --job-name=8B_scaling_${NUM_NODES}nodes \
           --export=ALL,\
CONFIG_FILE_PATH="$CONFIG_FILE_PATH",\
EXPERIMENT_FOLDER_PATH="$EXPERIMENT_FOLDER_PATH",\
PYTHON_ENV_PATH="$PYTHON_ENV_PATH",\
NUM_WARMUP_STEPS="$NUM_WARMUP_STEPS",\
NUM_MEASUREMENT_STEPS="$NUM_MEASUREMENT_STEPS" \
           submit_job.sbatch
done
