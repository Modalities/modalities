#!/bin/sh
set -eu

# change to the script's directory
# this ensures that relative paths work correctly
cd "$(dirname "$0")" || exit 1

# --- Config ---
EXPERIMENT_ROOT=$1 # The first argument is the experiment root directory
CONFIG_LIST_FILE="config_file_list.txt"
EXPECTED_STEPS=20

# --- Functions ---

# Extracts the last integer-named folder from the given file path
get_last_integer_folder() {
    path="$1"
    dir=$(dirname "$path")

    while [ "$dir" != "/" ]; do
        base=$(basename "$dir")
        case "$base" in
            ''|*[!0-9]*) ;;
            *) echo "$base"; return 0 ;;
        esac
        dir=$(dirname "$dir")
    done

    echo "Error: No integer-named folder found in path: $path" >&2
    return 1
}

# --- Step 1: Find configs to run ---
if [ -f "$CONFIG_LIST_FILE" ]; then
    echo "Removing existing $CONFIG_LIST_FILE"
    rm "$CONFIG_LIST_FILE"
fi

modalities benchmark list_remaining_runs --exp_root $EXPERIMENT_ROOT  --file_list_path $CONFIG_LIST_FILE --expected_steps $EXPECTED_STEPS

# --- Step 2: Loop over each config ---
while IFS= read -r file; do
    echo "Processing config: $file"

    num_ranks=$(get_last_integer_folder "$file")
    echo "  -> Using $num_ranks ranks"

    # Run your job
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((num_ranks - 1)))
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --rdzv-endpoint localhost:29504 --nnodes 1 --nproc_per_node 2 $(which modalities) run --config_file_path "$file" --experiment_id ""
done < "$CONFIG_LIST_FILE"

rm "$CONFIG_LIST_FILE"
echo "All configs processed."
