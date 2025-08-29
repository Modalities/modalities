#!/bin/sh
set -e 

cd "$(dirname "$0")" || exit 1

echo "Creating sweep configs..."
sh create_sweep_configs.sh

last_experiment_folder=$(realpath "$(ls -d ../experiments/* | sort | tail -n 1)")

echo "Running sweep: $last_experiment_folder"
sh single_node/run_single_node.sh "$last_experiment_folder"

rm -rf "$last_experiment_folder"