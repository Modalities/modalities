"""Job templates for different HPC systems."""

# This template is designed for MN5 at the Barcelona Supercomputing Center (BSC).
MN5_JOB_TEMPLATE = """#!/bin/bash
#SBATCH --exclusive
#SBATCH --account=ehpc17
#SBATCH --qos={}
#SBATCH --job-name={}
#SBATCH --chdir=/gpfs/projects/ehpc17/modalities
#SBATCH --output={}
#SBATCH --error={}
#SBATCH --nodes={}
#SBATCH --cpus-per-task=80
#SBATCH --ntasks {}
#SBATCH --gres=gpu:4
#SBATCH --time=00:20:00

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

#### Environment variables ####
export CXX=g++
export CC=gcc
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=50
export UCX_RC_TIMEOUT=4s
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export NCCL_IB_RETRY_CNT=10
export CUDA_VISIBLE_DEVICES=0,1,2,3

export MASTER_PORT=6000
export DEVICES_PER_NODE=4
export NUM_NODES="$SLURM_JOB_NUM_NODES"
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

set -euo pipefail
set -x

module load impi intel hdf5 mkl
module load  python/3.11.5-gcc
module load cuda/12.3
source {}

echo "START TIME: $(date)"

srun torchrun \
    --node_rank=$SLURM_PROCID \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --nnodes $NUM_NODES \
    --nproc_per_node 4 \
    --rdzv_backend c10d \
    $(which modalities) run \
    --config_file_path {}

echo "END TIME: $(date)"

echo "=== FINISHED ==="
"""

# This template is designed for the Leonardo system at CINECA.
LEONARDO_JOB_TEMPLATE = """#!/bin/bash
# SLURM SUBMIT SCRIPT
#SBATCH --exclusive
#SBATCH --account=p_gptx
#SBATCH --qos={}
#SBATCH --job-name={}
#SBATCH --output={}
#SBATCH --error={}
#SBATCH --time=00:30:00
#SBATCH --ntasks-per-node=4 
#SBATCH --nodes={}{}
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:4
#SBATCH --mem=0      
 
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
export CUDA_VISIBLE_DEVICES=0,1,2,3
 
# Enable logging
set -x -e
echo "START TIME: $(date)"
module load cuda/12.3
source {}
 
##### Network parameters #####
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

echo "START TIME: $(date)"

srun torchrun \
  --node_rank=$SLURM_PROCID \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT  \
  --nnodes $SLURM_JOB_NUM_NODES \
  --nproc_per_node 4 \
  --rdzv_backend c10d  \
  $(which modalities) run \
  --config_file_path {}

echo "END TIME: $(date)"
echo "=== FINISHED ==="

"""

# This template is designed for the Capella system at TU Dresden (ZIH)
CAPELLA_JOB_TEMPLATE = """#!/bin/bash
#SBATCH --exclusive
#SBATCH --account=p_gptx
#SBATCH --partition=capella 
#SBATCH --qos={}
#SBATCH --job-name={}
#SBATCH --output={}
#SBATCH --error={}
#SBATCH --nodes={}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=0  # Use all available memory on the node
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00

#### Environment variables ####
export CXX=g++
export CC=gcc
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=50
export UCX_RC_TIMEOUT=4s
export NCCL_IB_RETRY_CNT=10
export DEVICES_PER_NODE=4
export NUM_NODES="$SLURM_JOB_NUM_NODES"

set -euo pipefail
set -x

module load release/24.04  
module load GCC/13.2.0  
module load CUDA/12.4.0
ml Python/3.11.5
source {}

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

echo "START TIME: $(date)"

srun torchrun \
    --node_rank=$SLURM_PROCID \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --nnodes $NUM_NODES \
    --nproc_per_node 4 \
    --rdzv_backend c10d \
    $(which modalities) run \
    --config_file_path {}

echo "END TIME: $(date)"

echo "=== FINISHED ==="
"""
