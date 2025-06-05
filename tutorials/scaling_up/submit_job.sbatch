#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_D21_101
#SBATCH --qos=normal # boost_qos_dbg or normal
#SBATCH --job-name=8B_scaling
#SBATCH --output=logs/logs_%j.out
#SBATCH --error=logs/logs_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --gres=gpu:4
#SBATCH --time=5:00:00

#### Environment variables ####
export CXX=g++
export CC=gcc

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=50
export UCX_RC_TIMEOUT=4s
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export NCCL_IB_RETRY_CNT=10
export DEVICES_PER_NODE=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

set -euo pipefail
set -x

module load cuda/12.3

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6003

echo "START TIME: $(date)"
echo "Running on $SLURM_JOB_NUM_NODES nodes."

# Check required environment variables
: "${CONFIG_FILE_PATH:?Missing CONFIG_FILE_PATH}"
: "${EXPERIMENT_FOLDER_PATH:?Missing EXPERIMENT_FOLDER_PATH}"
: "${PYTHON_ENV_PATH:?Missing PYTHON_ENV_PATH}"
: "${NUM_WARMUP_STEPS:?Missing NUM_WARMUP_STEPS}"
: "${NUM_MEASUREMENT_STEPS:?Missing NUM_MEASUREMENT_STEPS}"

# Activate the Python environment
source "$PYTHON_ENV_PATH"

mkdir -p "$EXPERIMENT_FOLDER_PATH"

srun python scaling_grid_search.py \
    --config_file "$CONFIG_FILE_PATH" \
    --num_nodes "$SLURM_JOB_NUM_NODES" \
    --num_warmup_steps "$NUM_WARMUP_STEPS" \
    --num_measurement_steps "$NUM_MEASUREMENT_STEPS" \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --experiment_folder "$EXPERIMENT_FOLDER_PATH"

echo "END TIME: $(date)"
echo "=== FINISHED ==="
