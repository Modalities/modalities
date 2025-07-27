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
set -x  # Every command and its arguments are printed to stderr before being executed.

# Setup environment
module load cuda/12.3
module load gcc/12.2.0
module load binutils/2.42
source /leonardo_scratch/fast/EUHPC_D21_101/max_lue/python_envs/working/leonardo_modalities/bin/activate # TODO replace "working" with "stable" if needed

#### Environment variables
export CXX=g++
export CC=gcc
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

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=6015

# Get GPUs per node from Slurm or fallback
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-4}
NUM_RANKS=$((SLURM_JOB_NUM_NODES * GPUS_PER_NODE))
CONFIG_LIST_FILE="global_file_list.${NUM_RANKS}.${SLURM_PROCID}.txt"

echo "START TIME: $(date)"

# Step 1: Filter configs
modalities benchmark list_remaining_runs \
  --experiment_dir "$EXPERIMENT_ROOT/$NUM_RANKS" \
  --file_list_path "$CONFIG_LIST_FILE" \
  --expected_steps "$EXPECTED_STEPS" \
  --skip_exception_types "OutOfMemoryError,ValueError"


# Step 2: Read configs into Bash array and loop
mapfile -t config_files < "$CONFIG_LIST_FILE"

for config_file in "${config_files[@]}"; do
  echo "Processing config: $config_file"
  error_log_folder="$(dirname "$config_file")/error_logs"

  srun torchrun \
    --rdzv-endpoint "$MASTER_ADDR:$MASTER_PORT" \
    --nnodes "$SLURM_JOB_NUM_NODES" \
    --nproc_per_node "$GPUS_PER_NODE" \
    --rdzv_backend c10d \
    "$(which modalities)" run \
    --config_file_path "$config_file" \
    --experiment_id "" \
    --error_log_folder "$error_log_folder"
done

echo "END TIME: $(date)"

# Clean up
rm "$CONFIG_LIST_FILE"
echo "=== FINISHED ==="