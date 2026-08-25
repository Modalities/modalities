#!/bin/bash
# Check what is actually on disk, not what the pipeline believes it wrote.
#
# Three stages, each of which has caught something real:
#   scan_pbins   -- one file in 54,738 was 151 MB on disk reporting data_len=0
#   verify_blend -- packed tokens vs estimates, and packed documents vs documents selected
#   load_blend   -- WeightedCombinedDataset over every packed file, the path training takes
#SBATCH --job-name=q_verify
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
# Holding 54,738 memory-mapped datasets and one document index at a time.
#SBATCH --mem=220G
#SBATCH --time=08:00:00
#SBATCH --output=/home/richard.rutmann/logs/quality/verify_%j.out
#SBATCH --error=/home/richard.rutmann/logs/quality/verify_%j.err

set -uo pipefail

MQ="${MQ:-/data/user/richard.rutmann/venvs/modalities-quality/bin/python}"
QDIR="${QDIR:-/home/richard.rutmann/repos/modalities/config_files/data_preparation/quality}"
WORK="${WORK:?WORK is not set}"
SOURCE_ROOT="${SOURCE_ROOT:-/data/annealing}"

echo "=== 1. packed file headers ==="
srun "$MQ" "$QDIR/slurm/scan_pbins.py" --work_dir "$WORK" --out "$WORK/bad_pbins.txt"
SCAN=$?

echo "=== 2. tokens and document counts against the manifest ==="
srun "$MQ" "$QDIR/slurm/verify_blend.py" --work_dir "$WORK" --source_root "$SOURCE_ROOT"
VERIFY=$?

echo "=== 3. the blend loads as training will load it ==="
srun "$MQ" "$QDIR/slurm/load_blend.py" --work_dir "$WORK"
LOAD=$?

echo "scan=$SCAN verify=$VERIFY load=$LOAD"
exit $(( SCAN | VERIFY | LOAD ))
