#!/bin/bash
# Aggregate the joined sidecars into cubes. Must run after every join task has finished.
# Single task; ~50 min for the whole blend, and it holds a few GB while grouping.
#SBATCH --job-name=q_cubes
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/home/richard.rutmann/logs/quality/3b_cubes_%j.out
#SBATCH --error=/home/richard.rutmann/logs/quality/3b_cubes_%j.err

set -euo pipefail

MQ="${MQ:-/data/user/richard.rutmann/venvs/modalities-quality/bin/python}"
QDIR="${QDIR:-/home/richard.rutmann/repos/modalities/config_files/data_preparation/quality}"
WORK="${WORK:?WORK is not set}"
REGISTRY="${REGISTRY:-$QDIR/annealing_registry.yaml}"

unset SLURM_MEM_PER_CPU || true
unset SLURM_MEM_PER_GPU || true

echo "START $(date)"
srun "$MQ" -m modalities quality build-cube --registry "$REGISTRY" --work_dir "$WORK"
echo "Coverage per dataset: $WORK/join_report.json -- read it before trusting a selection."
echo "END $(date)"
