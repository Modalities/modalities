#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=8B_FSDP2_compiled
#SBATCH --output=../../../logs/8B_large_scale_%j.out
#SBATCH --error=../../../logs/8B_large_scale_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0               # Use all available memory on the node
#SBATCH --gres=gpu:4
#SBATCH --exclusive

# Enable logging
set -euo pipefail
set -x

# Setup environment
module load cuda/12.3
source /leonardo_scratch/fast/EUHPC_D21_101/max_lue/python_envs/working/leonardo_modalities/bin/activate # TODO replace working with stable

#### Environment variables
export CXX=g++
export CC=gcc
# chatgpt suggested env variables
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=100
export NCCL_SOCKET_TIMEOUT=3600
export UCX_RC_TIMEOUT=10s
export UCX_TLS=rc
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT
export NCCL_IB_RETRY_CNT=10
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export CUDA_VISIBLE_DEVICES=0,1,2,3

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=$((6000 + RANDOM % 1000))  # randomize to avoid reuse collisions

# Get GPUs per node from Slurm or fallback
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-4}
# Compute total number of ranks
NUM_RANKS=$((SLURM_JOB_NUM_NODES * GPUS_PER_NODE))
# Construct config list filename
CONFIG_LIST_FILE="global_file_list.${NUM_RANKS}.txt"

echo "START TIME: $(date)"
# --- Step 1: Filter the remaining runs on a single process while others wait (prevents race conditions) ---
srun --ntasks=1 modalities benchmark list_missing_runs --experiment_dir $EXPERIMENT_ROOT  --file_list_path $CONFIG_LIST_FILE --expected_steps $EXPECTED_STEPS

# --- Step 2: Loop over each remaining config and benchmark the setting ---
while IFS= read -r config_file; do
    echo "Processing config: $config_file"
    srun torchrun --rdzv-endpoint $MASTER_ADDR:$MASTER_PORT --nnodes $SLURM_JOB_NUM_NODES --nproc_per_node $GPUS_PER_NODE $(which modalities) run --config_file_path $config_file --experiment_id ""
done < "$CONFIG_LIST_FILE"
echo "END TIME: $(date)"


# Clean up
srun --ntasks=1 rm "$CONFIG_LIST_FILE"


echo "=== FINISHED ==="
