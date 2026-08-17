#!/bin/bash
# Tokenize the selected documents. Each config points at a filtered index, so only the
# documents that survived the selection are read and tokenized.
#
# Set the array upper bound to (number of configs / PACK_CONFIGS_PER_TASK) - 1:
#   wc -l < $WORK/packcfg_list.txt
#SBATCH --job-name=q_pack
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=/home/richard.rutmann/logs/quality/4_pack_%A_%a.out
#SBATCH --error=/home/richard.rutmann/logs/quality/4_pack_%A_%a.err
#SBATCH --array=0-63

set -euo pipefail

MQ="${MQ:-/data/user/richard.rutmann/venvs/modalities-quality/bin/python}"
WORK="${WORK:?WORK is not set}"
CONFIG_LIST="${CONFIG_LIST:-$WORK/packcfg_list.txt}"
PER_TASK="${PACK_CONFIGS_PER_TASK:-1}"

export HF_HOME="${HF_HOME:-/data/cache/hf_cache}"
unset SLURM_MEM_PER_CPU || true
unset SLURM_MEM_PER_GPU || true

START=$((SLURM_ARRAY_TASK_ID * PER_TASK + 1))
END=$((START + PER_TASK - 1))
echo "START $(date) configs ${START}..${END} of $(wc -l < "$CONFIG_LIST")"

sed -n "${START},${END}p" "$CONFIG_LIST" | while read -r cfg; do
  [ -n "$cfg" ] || continue
  echo "--- packing $cfg"
  srun "$MQ" -m modalities data pack_encoded_data "$cfg" --file_existence_policy skip
done

echo "END $(date)"
