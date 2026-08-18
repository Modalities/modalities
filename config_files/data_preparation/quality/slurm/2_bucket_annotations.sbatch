#!/bin/bash
# Partition the propella splits by a hash of the join key. The expensive half of the
# join: HPLT alone is ~12 bn annotation rows. Each task writes its own file per bucket.
#SBATCH --job-name=q_bucket
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
# As in the sidecar script: the CPU request is there to limit how many bandwidth-hungry
# tasks SLURM packs onto one node, not because a task needs 8 cores.
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/home/richard.rutmann/logs/quality/2_bucket_%A_%a.out
#SBATCH --error=/home/richard.rutmann/logs/quality/2_bucket_%A_%a.err
#SBATCH --array=0-63

set -euo pipefail

MQ="${MQ:-/data/user/richard.rutmann/venvs/modalities-quality/bin/python}"
QDIR="${QDIR:-/home/richard.rutmann/repos/modalities/config_files/data_preparation/quality}"
WORK="${WORK:?WORK is not set}"
NUM_BUCKETS="${NUM_BUCKETS:-1024}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset SLURM_MEM_PER_CPU || true
unset SLURM_MEM_PER_GPU || true

NUM_SHARDS="${SLURM_ARRAY_TASK_COUNT:-64}"
echo "START $(date) shard ${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS}"

# Every task in the array must finish. join-annotations refuses to run against an
# incomplete bucketing run rather than silently dropping a missing task's annotations,
# so a failed task is a hard stop, not a quiet gap. Re-run just that shard id.
srun "$MQ" -m modalities quality bucket-annotations \
  --registry "${REGISTRY:-$QDIR/annealing_registry.yaml}" \
  --work_dir "$WORK" \
  --num_buckets "$NUM_BUCKETS" \
  --shard_id "$SLURM_ARRAY_TASK_ID" \
  --num_shards "$NUM_SHARDS"

echo "END $(date)"
