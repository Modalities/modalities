#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_D21_101
#SBATCH --qos=normal
#SBATCH --job-name=15B_FSDP2_compiled
#SBATCH --output=/leonardo_scratch/fast/EUHPC_D21_101/max_lue/repositories/working/modalities/tutorials/scaling_up/logs/logs_%j.out
#SBATCH --error=/leonardo_scratch/fast/EUHPC_D21_101/max_lue/repositories/working/modalities/tutorials/scaling_up/logs/logs_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0  # Use all available memory on the node
#SBATCH --gres=gpu:4
#SBATCH --time=00:20:00

#### Environment variables ####
export CXX=g++
export CC=gcc

# force crashing on nccl issues like hanging broadcast
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=50
export UCX_RC_TIMEOUT=4s
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export NCCL_IB_RETRY_CNT=10
export DEVICES_PER_NODE=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Enable logging
set -euo pipefail
set -x

module load cuda/12.3

source /leonardo_scratch/fast/EUHPC_D21_101/max_lue/python_envs/working/leonardo_modalities/bin/activate

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6003

echo "START TIME: $(date)"
echo "Running on $SLURM_JOB_NUM_NODES nodes."

CONFIG_FILE_PATH="config_8B_scaling.yaml"
EXPERIMENT_FOLDER_PATH="experiments/$DATE_OF_RUN"
mkdir -p "$EXPERIMENT_FOLDER_PATH"

srun python scaling_grid_search.py \
    --config_file $CONFIG_FILE_PATH \
    --num_nodes $SLURM_JOB_NUM_NODES \
    --num_warmup_steps 16 \
    --num_measurement_steps 16 \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --experiment_folder $EXPERIMENT_FOLDER_PATH

echo "END TIME: $(date)"
echo "=== FINISHED ==="
