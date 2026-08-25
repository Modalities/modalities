#!/bin/bash
# Tokenize the selected documents. Each config points at a filtered index, so only the
# documents that survived the selection are read and tokenized.
#
# Driven through pack_many.py rather than one `modalities data pack_encoded_data` call per
# config. That CLI rebuilds its components, tokenizer included, on every invocation: 24.7 s
# measured, against ~3 s of real work for a Dolmino file. The real blend renders 54,738
# configs, so per-config invocation spends ~375 core-hours loading the tokenizer to do about
# 48 core-hours of tokenising. Inside the driver the load costs 1.2 s and is paid once.
#
# Measured on the full annealing blend: 1 h 40 m for 6.0 TB of output, zero failures.
#SBATCH --job-name=q_pack
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
# All of a node's cores: the packer spawns workers from the node CPU count, so one task per
# node avoids oversubscription.
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=/home/richard.rutmann/logs/quality/pack_%A_%a.out
#SBATCH --error=/home/richard.rutmann/logs/quality/pack_%A_%a.err
#SBATCH --array=0-63

set -euo pipefail

MQ="${MQ:-/data/user/richard.rutmann/venvs/modalities-quality/bin/python}"
QDIR="${QDIR:-/home/richard.rutmann/repos/modalities/config_files/data_preparation/quality}"
WORK="${WORK:?WORK is not set}"
CONFIG_LIST="${CONFIG_LIST:-$WORK/packcfg_list.txt}"
export HF_HOME="${HF_HOME:-/data/cache/hf_cache}"
unset SLURM_MEM_PER_CPU || true
unset SLURM_MEM_PER_GPU || true

NUM_SHARDS="${SLURM_ARRAY_TASK_COUNT:-64}"
echo "START $(date) shard ${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS}"

# --skip_existing checks each output's header, not merely that the file is there: a .pbin
# left by an interrupted run can be megabytes on disk and still report data_len=0.
srun "$MQ" "$QDIR/slurm/pack_many.py" \
  --config_list "$CONFIG_LIST" \
  --shard_id "$SLURM_ARRAY_TASK_ID" \
  --num_shards "$NUM_SHARDS" \
  --tokenizer_config "${TEMPLATE:-$QDIR/annealing_packing_template.yaml}" \
  --skip_existing

echo "END $(date)"
