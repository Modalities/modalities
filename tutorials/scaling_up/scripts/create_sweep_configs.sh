#!/bin/sh
set -eu
# change to the script's directory
# this ensures that relative paths work correctly
cd "$(dirname "$0")" || exit 1

modalities benchmark prepare_sweep_configs --sweep_config_path ../configs/sweep_config.yaml --output_dir ../experiments --world_sizes 2,4,8