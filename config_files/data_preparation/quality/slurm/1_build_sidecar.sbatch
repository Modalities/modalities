#!/bin/bash
# Per-document sidecar over /data/annealing. The only stage that reads the raw data.
# Work is divided per JSONL file across every enabled dataset, so this one array covers
# the whole blend regardless of how unevenly file counts fall.
#SBATCH --job-name=q_sidecar
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
# 8 CPUs is not what a task needs -- it is single-threaded. It is what stops SLURM
# packing 16 of them onto one node. The binding constraint is per-node bandwidth to
# /data, not CPU: a real run put 12 tasks on one node and each got 95 MB/s against the
# 282 MB/s a single stream gets, while four nodes sat idle. At 8 CPUs only 4 tasks fit
# per node, so an array of 64 spreads over 16 nodes instead of piling onto 4.
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/home/richard.rutmann/logs/quality/1_sidecar_%A_%a.out
#SBATCH --error=/home/richard.rutmann/logs/quality/1_sidecar_%A_%a.err
#SBATCH --array=0-63

set -euo pipefail

MQ="${MQ:-/data/user/richard.rutmann/venvs/modalities-quality/bin/python}"
QDIR="${QDIR:-/home/richard.rutmann/repos/modalities/config_files/data_preparation/quality}"
WORK="${WORK:?WORK is not set}"

# Each task is single-threaded and I/O-bound at ~374 MB/s where bandwidth allows. The
# 3.8 GB/s aggregate measured on the login node is not what a compute node gets, so
# spreading tasks across nodes matters more than raising the array size.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset SLURM_MEM_PER_CPU || true
unset SLURM_MEM_PER_GPU || true

NUM_SHARDS="${SLURM_ARRAY_TASK_COUNT:-64}"

# ONLY_DATASETS restricts the run to a space-separated list of names. Unset, the stage covers
# every enabled dataset as before. It matters when adding datasets to an existing work
# directory: without it the array would rebuild the sidecars of datasets that already have
# good ones, which is both hours of wasted reading and an overwrite of verified work.
# SIDECAR_RESUME=1 skips files whose part already reads as valid parquet. Use it when
# re-running after a partial failure; leave it unset after changing a native metric, since
# the existing parts are then stale.
ONLY_ARGS=""
if [[ -n "${ONLY_DATASETS:-}" ]]; then
    for _d in $ONLY_DATASETS; do ONLY_ARGS="$ONLY_ARGS --only $_d"; done
fi

echo "START $(date) shard ${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS}${ONLY_DATASETS:+ restricted to: $ONLY_DATASETS}"

srun "$MQ" -m modalities quality build-sidecar \
  --registry "${REGISTRY:-$QDIR/annealing_registry.yaml}" \
  --work_dir "$WORK" \
  --index_root "$WORK/idx" \
  --shard_id "$SLURM_ARRAY_TASK_ID" \
  --num_shards "$NUM_SHARDS" \
  ${SIDECAR_RESUME:+--resume} \
  $ONLY_ARGS

echo "END $(date)"
