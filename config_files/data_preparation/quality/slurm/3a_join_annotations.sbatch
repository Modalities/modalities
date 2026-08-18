#!/bin/bash
# Attach the bucketed annotations to each dataset's sidecar. One array task per annotated
# dataset: the joins are independent, so wall time is the slowest dataset (~2 h for
# nemotron-cc) rather than the sum (~6 h).
#
# Set the array upper bound to (number of annotated datasets) - 1. Print the list with:
#   $MQ -c "from modalities.dataloader.preprocessing.quality.registry import CorpusRegistry; \
#           r=CorpusRegistry.from_yaml('$QDIR/annealing_registry.yaml'); \
#           print([d.name for d in r.enabled_datasets() if d.annotation_split])"
#SBATCH --job-name=q_join
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
# 8 CPUs to cap tasks per node: this is bandwidth-bound, not CPU-bound. See the README.
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/richard.rutmann/logs/quality/3a_join_%A_%a.out
#SBATCH --error=/home/richard.rutmann/logs/quality/3a_join_%A_%a.err
#SBATCH --array=0-15

set -euo pipefail

MQ="${MQ:-/data/user/richard.rutmann/venvs/modalities-quality/bin/python}"
QDIR="${QDIR:-/home/richard.rutmann/repos/modalities/config_files/data_preparation/quality}"
WORK="${WORK:?WORK is not set}"
REGISTRY="${REGISTRY:-$QDIR/annealing_registry.yaml}"

unset SLURM_MEM_PER_CPU || true
unset SLURM_MEM_PER_GPU || true

# Resolve this task's dataset from the registry, so the mapping cannot drift from the
# config the way a hand-maintained list would.
DATASET=$("$MQ" - "$REGISTRY" "$SLURM_ARRAY_TASK_ID" <<'PY'
import sys
from modalities.dataloader.preprocessing.quality.registry import CorpusRegistry
reg = CorpusRegistry.from_yaml(sys.argv[1])
names = [d.name for d in reg.enabled_datasets() if d.annotation_split]
idx = int(sys.argv[2])
print(names[idx] if idx < len(names) else "")
PY
)

if [[ -z "$DATASET" ]]; then
    echo "array index $SLURM_ARRAY_TASK_ID is past the last annotated dataset; nothing to do"
    exit 0
fi

echo "START $(date) dataset=$DATASET"
srun "$MQ" -m modalities quality join-annotations \
    --registry "$REGISTRY" --work_dir "$WORK" --only "$DATASET"
echo "END $(date)"
