#!/bin/bash
# Attach the annotations to the sidecars, then aggregate into cubes. Needs the sidecar
# and bucketing arrays finished. Single task; the cube build is the longer half (~50 min
# for the full blend) and holds a few GB while grouping.
#SBATCH --job-name=q_join_cube
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/richard.rutmann/logs/quality/3_join_cube_%j.out
#SBATCH --error=/home/richard.rutmann/logs/quality/3_join_cube_%j.err

set -euo pipefail

MQ="${MQ:-/data/user/richard.rutmann/venvs/modalities-quality/bin/python}"
QDIR="${QDIR:-/home/richard.rutmann/repos/modalities/config_files/data_preparation/quality}"
WORK="${WORK:?WORK is not set}"

unset SLURM_MEM_PER_CPU || true
unset SLURM_MEM_PER_GPU || true
echo "START $(date)"

srun "$MQ" -m modalities quality join-annotations \
  --registry "${REGISTRY:-$QDIR/annealing_registry.yaml}" --work_dir "$WORK"

srun "$MQ" -m modalities quality build-cube \
  --registry "${REGISTRY:-$QDIR/annealing_registry.yaml}" --work_dir "$WORK"

echo "Coverage per dataset is in $WORK/join_report.json -- read it before trusting a selection."
echo "END $(date)"
