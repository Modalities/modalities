#!/bin/bash
# Check what is actually on disk, not what the pipeline believes it wrote.
#
# One stage now that the output is JSONL rather than packed tokens:
#   verify_jsonl -- every dataset's line count against its record, the realised
#                   lines-per-document against the ratio that was asked for, a byte-for-byte
#                   replay of sampled shards against the corpus, and a check that nothing was
#                   written into the source tree.
#
# The replay is the one that would catch a real problem: it recomputes which documents the
# export should have written and how many times, and compares that to the bytes on disk.
#SBATCH --job-name=q_verify
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
# Streams shards and source files a line at a time; nothing accumulates.
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/richard.rutmann/logs/quality/verify_%j.out
#SBATCH --error=/home/richard.rutmann/logs/quality/verify_%j.err

set -uo pipefail

MQ="${MQ:-/data/user/richard.rutmann/venvs/modalities-quality/bin/python}"
QDIR="${QDIR:-/home/richard.rutmann/repos/modalities/config_files/data_preparation/quality}"
WORK="${WORK:?WORK is not set}"
SOURCE_ROOT="${SOURCE_ROOT:-/data/annealing}"
# Re-counting every line reads the whole 9.75 TB export; off by default, worth it once.
COUNT_LINES="${COUNT_LINES:-}"

srun "$MQ" "$QDIR/slurm/verify_jsonl.py" \
    --manifest "$WORK/out/export_manifest.yaml" \
    --mix_manifest "$WORK/mix/mix_manifest.yaml" \
    --source_root "$SOURCE_ROOT" \
    --shards 8 ${COUNT_LINES:+--count_lines}
