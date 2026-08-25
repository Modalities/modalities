#!/bin/bash
# Write the selected documents out as JSONL, one shard per source file, with the up- and
# downsampling materialised in the bytes. One array task per dataset: the datasets are
# independent, so wall time is the largest one rather than the sum.
#
# Set the array upper bound to (number of datasets in the mix manifest) - 1. Print the list:
#   $MQ -c "import yaml,sys; m=yaml.safe_load(open('$WORK/mix/mix_manifest.yaml')); \
#           print(sorted({d.get('source_dataset') or d['name'] for d in m['datasets']}))"
#
# Each task writes its own dataset record; the blend-wide export_manifest.yaml is merged
# afterwards by the --finalize_only step below, because eighteen tasks writing one shared
# manifest would race and the last writer would erase the rest.
#SBATCH --job-name=q_export
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
# Pure byte copying: read the selected lines, write them out. Bandwidth-bound, not CPU-bound,
# so 8 CPUs is here to cap tasks per node rather than because they are used.
#SBATCH --cpus-per-task=8
# One source file's index at a time plus an mmap of that file. Nothing accumulates across
# files, so this is far below the join or the apply stage.
#SBATCH --mem=32G
# Estimated 9.75 TB written in total, dominated by nemotron-cc at 2.14 TB and klettermix-de
# at 1.68 TB -- both inflated by their ratios. Packing 6.0 TB took 1 h 40 m with tokenisation;
# this does less work per byte but writes more of them.
#SBATCH --time=12:00:00
#SBATCH --output=/home/richard.rutmann/logs/quality/4_export_%A_%a.out
#SBATCH --error=/home/richard.rutmann/logs/quality/4_export_%A_%a.err
#SBATCH --array=0-17

set -euo pipefail

MQ="${MQ:-/data/user/richard.rutmann/venvs/modalities-quality/bin/python}"
QDIR="${QDIR:-/home/richard.rutmann/repos/modalities/config_files/data_preparation/quality}"
WORK="${WORK:?WORK is not set}"
REGISTRY="${REGISTRY:-$QDIR/annealing_registry.yaml}"
MANIFEST="${MANIFEST:-$WORK/mix/mix_manifest.yaml}"
OUT="${OUT:-$WORK/out}"

unset SLURM_MEM_PER_CPU || true
unset SLURM_MEM_PER_GPU || true

# Resolved from the manifest rather than a hand-maintained list, so the mapping cannot drift
# from what was actually materialised. Curve buckets collapse to their source dataset.
DATASET=$("$MQ" - "$MANIFEST" "$SLURM_ARRAY_TASK_ID" <<'PY'
import sys, yaml
m = yaml.safe_load(open(sys.argv[1]))
names = sorted({d.get("source_dataset") or d["name"] for d in m["datasets"]})
idx = int(sys.argv[2])
print(names[idx] if idx < len(names) else "")
PY
)

if [[ -z "$DATASET" ]]; then
    echo "array index $SLURM_ARRAY_TASK_ID is past the last dataset; nothing to do"
    exit 0
fi

echo "START $(date) dataset=$DATASET"
srun "$MQ" -m modalities quality export-jsonl \
    --manifest "$MANIFEST" --registry "$REGISTRY" --output_dir "$OUT" \
    --only "$DATASET" --no_finalize
echo "END $(date)"
