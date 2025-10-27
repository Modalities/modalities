#!/bin/sh
set -eu

# change to the script's directory
# this ensures that relative paths work correctly
cd "$(dirname "$0")" || exit 1


# --- Config ---
EXPERIMENT_ROOT="../../../experiments/2025-09-25__16-30-29_9a5c69962be24784"   # TODO: Change to your experiment root
EXPECTED_STEPS=20
CONFIG_LIST_FILE="global_file_list.txt"

ACCOUNT=EUHPC_E05_119   # TODO: Change to your account
TIME_LIMIT=03:00:00
GPUS_PER_NODE=4

# Retrieve the list of configs to run
modalities benchmark list_remaining_runs --exp_root "$EXPERIMENT_ROOT" --file_list_path "$CONFIG_LIST_FILE" --expected_steps "$EXPECTED_STEPS" --skip_exception_types "OutOfMemoryError,ValueError"


worldsizes=$(awk -F'/' '{print $(NF-2)}' $CONFIG_LIST_FILE | sort -u)
for ws in $worldsizes; do

    # make sure that the world size is a multiple of GPUs per node
    if [ $(( ws % GPUS_PER_NODE )) -ne 0 ]; then
        echo "Error: world size ($ws) must be a multiple of GPUs per node ($GPUS_PER_NODE)." >&2
        exit 1
    fi


    # Calculate the number of nodes needed for the current world size
    NODES=$(( (ws + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))
    # select the appropriate QOS based on the number of nodes
    if [ "$NODES" -gt 64 ]; then
        QOS="boost_qos_bprod"   # TODO: change to the correct partition
    else
        QOS="normal"            # TODO: change to the correct partition
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