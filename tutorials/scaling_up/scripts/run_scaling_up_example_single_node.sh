#!/bin/sh
set -e

cd "$(dirname "$0")" || exit 1

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

echo "Creating sweep configs..."
sh create_sweep_configs.sh

last_experiment_folder=$(realpath "$(ls -d ../experiments/* | sort | tail -n 1)")

echo "Running sweep: $last_experiment_folder"
sh single_node/run_single_node.sh "$last_experiment_folder"

if [ "$remove_last_experiment_folder" = true ]; then
    echo "Cleaning up: removing last experiment folder: $last_experiment_folder"
    rm -rf "$last_experiment_folder"
fi
