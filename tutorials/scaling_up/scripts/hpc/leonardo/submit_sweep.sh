#!/bin/sh
set -eu

# change to the script's directory
# this ensures that relative paths work correctly
cd "$(dirname "$0")" || exit 1


# --- Config ---
EXPERIMENT_ROOT="../../experiments/2025-07-30__21-50-11_fca0790e"
EXPECTED_STEPS=20
CONFIG_LIST_FILE="global_file_list.txt"

ACCOUNT=EUHPC_E05_119
TIME_LIMIT=03:00:00
GPUS_PER_NODE=4

# Retrieve the list of configs to run
modalities benchmark list_remaining_runs --experiment_dir "$EXPERIMENT_ROOT" --file_list_path "$CONFIG_LIST_FILE" --expected_steps "$EXPECTED_STEPS" --skip_exception_types "OutOfMemoryError,ValueError"


worldsizes=$(awk -F'/' '{print $(NF-2)}' $CONFIG_LIST_FILE | sort -u)
for ws in $worldsizes; do
    # Calculate the number of nodes needed for the current world size
    NODES=$(( (ws + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))
    # select the appropriate QOS based on the number of nodes
    if [ "$NODES" -gt 64 ]; then
        QOS="boost_qos_bprod"
    else
        QOS="normal" # "boost_qos_dbg" # normal
    fi
    # submit the job for the current world size
    sbatch --account=$ACCOUNT \
        --qos=$QOS \
        --nodes=$NODES \
        --time=$TIME_LIMIT \
        --export=EXPERIMENT_ROOT=$EXPERIMENT_ROOT,EXPECTED_STEPS=$EXPECTED_STEPS  \
        job.sbatch
done

rm "$CONFIG_LIST_FILE"
echo "All jobs submitted."