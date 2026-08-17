#!/bin/bash
# Per-document sidecar over /data/annealing. The only stage that reads the raw data.
# Work is divided per JSONL file across every enabled dataset, so this one array covers
# the whole blend regardless of how unevenly file counts fall.
#SBATCH --job-name=q_sidecar
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/home/richard.rutmann/logs/quality/1_sidecar_%A_%a.out
#SBATCH --error=/home/richard.rutmann/logs/quality/1_sidecar_%A_%a.err
#SBATCH --array=0-63

set -euo pipefail

MQ="${MQ:-/data/user/richard.rutmann/venvs/modalities-quality/bin/python}"
QDIR="${QDIR:-/home/richard.rutmann/repos/modalities/config_files/data_preparation/quality}"
WORK="${WORK:?WORK is not set}"

# Each task is single-threaded and I/O-bound at ~374 MB/s, against ~3.8 GB/s aggregate
# from /data. Past roughly 16 concurrent tasks the filesystem is the limit, not the code.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset SLURM_MEM_PER_CPU || true
unset SLURM_MEM_PER_GPU || true

NUM_SHARDS="${SLURM_ARRAY_TASK_COUNT:-64}"
echo "START $(date) shard ${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS}"

srun "$MQ" -m modalities quality build-sidecar \
  --registry "${REGISTRY:-$QDIR/annealing_registry.yaml}" \
  --work_dir "$WORK" \
  --index_root "$WORK/idx" \
  --shard_id "$SLURM_ARRAY_TASK_ID" \
  --num_shards "$NUM_SHARDS"

echo "END $(date)"
