#!/bin/sh

set -e

cd "$(dirname "$0")"

remove_last_experiment_folder=false

# Parse flags
# -c : remove the last experiment folder after running the sweep as a cleanup step
# this flag is useful when you just want to test the sweep (e.g., pytest) 
# and don't want to keep the generated experiment folder
while getopts "c" opt; do
  case "$opt" in
    c) remove_last_experiment_folder=true ;;
    *) echo "Usage: $0 [-c]"; exit 1 ;;
  esac
done

# Run torchrun
torchrun \
  --rdzv-endpoint localhost:29589 \
  --nnodes 1 \
  --nproc_per_node 4 \
  run_distributed_model_profiling.py


last_experiment_folder=$(realpath "$(ls -d ../../experiments/* | sort | tail -n 1)")


if [ "$remove_last_experiment_folder" = true ]; then
    echo "Cleaning up: removing last experiment folder: $last_experiment_folder"
    rm -rf "$last_experiment_folder"
fi