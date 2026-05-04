#!/bin/bash
# SLURM SUBMIT SCRIPT
#SBATCH --exclusive
#SBATCH --account=your_account
#SBATCH --partition=your_partition
#SBATCH --qos=your_qos
#SBATCH --job-name=modalities
#SBATCH --output=/path/to/logs/log_%j.out
#SBATCH --error=/path/to/logs/log_%j.err
#SBATCH --time=00:10:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --nodes=32
#SBATCH --gres=gpu:4
#SBATCH --mem=0

# Paths (set these to real host locations before submitting):
SINGULARITY_IMAGE=./modalities.sif                # Singularity image file.
CONTAINER_HOME=/path/to/container/home/on/host    # Acts as $HOME inside container (-H).
MODALITIES_DIR=/path/to/modalities/on/host        # Host repo path bind-mounted into container.

#### Environment variables ####
export CXX=g++
export CC=gcc

# NCCL/UCX settings
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=50
export UCX_RC_TIMEOUT=4s
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export NCCL_IB_RETRY_CNT=10
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1

# Enable logging
set -x -e
echo "START TIME: $(date)"

##### Network parameters #####
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

echo "START TIME: $(date)"

# Launch the job via srun; Singularity provides GPUs with --nv.
# Additional bind mounts (-B) can be added for datasets, scratch, etc.
srun singularity exec --nv \
  -H $CONTAINER_HOME \
  -B /dev/infiniband:/dev/infiniband \
  -B $MODALITIES_DIR:/opt/modalities \
  # bind other directories as needed, e.g. to access data on host system
  "$SINGULARITY_IMAGE" bash -lc "
    cd /opt/modalities
    torchrun \
      --node_rank=$SLURM_NODEID \
      --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
      --rdzv-id test_pp \
      --nnodes $SLURM_JOB_NUM_NODES \
      --nproc_per_node 4 \
      --rdzv_backend c10d \
      src/modalities/__main__.py run \
      --config_file_path config_files/training/config_lorem_ipsum_long_fsdp2.yaml \
      --test_comm
  "

echo "END TIME: $(date)"
echo "=== FINISHED ==="
