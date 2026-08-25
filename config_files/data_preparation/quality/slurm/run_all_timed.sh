#!/usr/bin/env bash
# Runs the quality pipeline end to end against /data/annealing, printing how long each
# step took and a summary at the end.
#
#   bash config_files/data_preparation/quality/slurm/run_all_timed.sh
#   bash .../run_all_timed.sh 1 2 3 4        # only these steps
#
# Array jobs are submitted with `sbatch --wait`, so each timing is the job's real
# duration rather than how long submission took. That also makes the steps run in order,
# which steps 4 and 6 onwards require.

set -uo pipefail

# ----------------------------------------------------------------------- settings
export MQ="${MQ:-/data/user/richard.rutmann/venvs/modalities-quality/bin/python}"
export REPO="${REPO:-/home/richard.rutmann/repos/modalities}"
export QDIR="${QDIR:-$REPO/config_files/data_preparation/quality}"
export WORK="${WORK:-/data/user/richard.rutmann/annealing_blend}"
export HF_HOME="${HF_HOME:-/data/cache/hf_cache}"

# Overridable so a blend variant can be run without editing the shared configs.
REGISTRY="${REGISTRY:-$QDIR/annealing_registry.yaml}"
SELECTION="${SELECTION:-$QDIR/annealing_selection.yaml}"
TOKENIZER_CONFIG="${TOKENIZER_CONFIG:-$QDIR/annealing_tokenizer.yaml}"

SIDECAR_TASKS="${SIDECAR_TASKS:-64}"
BUCKET_TASKS="${BUCKET_TASKS:-64}"
NUM_BUCKETS="${NUM_BUCKETS:-1024}"
PACK_TASKS="${PACK_TASKS:-64}"
SAMPLE_SIZE="${SAMPLE_SIZE:-2000}"
BLEND_NAME="${BLEND_NAME:-blend_v1}"

if [[ -f "$HOME/.config/huggingface/token" ]]; then
    export HF_TOKEN="$(tr -d '\r\n' < "$HOME/.config/huggingface/token")"
fi
mkdir -p "$WORK" "$HOME/logs/quality"
cd "$REPO"

WANTED=("$@")
want() {
    [[ ${#WANTED[@]} -eq 0 ]] && return 0
    local s
    for s in "${WANTED[@]}"; do [[ "$s" == "$1" ]] && return 0; done
    return 1
}

# ------------------------------------------------------------------------ timing
declare -a SUMMARY=()
OVERALL_START=$(date +%s)

fmt() {
    local s=$1
    printf '%dh %02dm %02ds' $((s / 3600)) $(((s % 3600) / 60)) $((s % 60))
}

step() {
    local id="$1" label="$2"; shift 2
    if ! want "$id"; then
        SUMMARY+=("$(printf '%-3s %-34s %14s' "$id" "$label" 'skipped')")
        return 0
    fi
    printf '\n===== step %s: %s\n===== started %s\n' "$id" "$label" "$(date '+%F %T')"
    local t0 rc dt
    t0=$(date +%s)
    "$@"
    rc=$?
    dt=$(( $(date +%s) - t0 ))
    printf '===== step %s took %s (exit %d)\n' "$id" "$(fmt "$dt")" "$rc"
    SUMMARY+=("$(printf '%-3s %-34s %14s  exit %d' "$id" "$label" "$(fmt "$dt")" "$rc")")
    if [[ $rc -ne 0 ]]; then
        echo "!!!!! step $id failed; stopping. Fix it and re-run with: bash $0 $id ..." >&2
        exit "$rc"
    fi
}

print_summary() {
    printf '\n%s\n' "=========================================================================="
    printf '%-3s %-34s %14s\n' "id" "step" "elapsed"
    printf '%s\n' "--------------------------------------------------------------------------"
    local line
    for line in "${SUMMARY[@]}"; do printf '%s\n' "$line"; done
    printf '%s\n' "--------------------------------------------------------------------------"
    printf '%-38s %14s\n' "TOTAL" "$(fmt $(( $(date +%s) - OVERALL_START )))"
    printf '%s\n' "=========================================================================="
}
trap print_summary EXIT

EXPORTS="ALL,MQ=$MQ,QDIR=$QDIR,WORK=$WORK,NUM_BUCKETS=$NUM_BUCKETS,HF_HOME=$HF_HOME,REGISTRY=$REGISTRY"

# ============================================================ once per blend
step 1 "calibrate tokens" \
    "$MQ" -m modalities quality calibrate \
        --registry "$REGISTRY" --work_dir "$WORK" \
        --tokenizer_config "$TOKENIZER_CONFIG" --sample_size "$SAMPLE_SIZE"

step 2 "build sidecar (array)" \
    sbatch --wait --export="$EXPORTS" --array="0-$((SIDECAR_TASKS - 1))" \
        "$QDIR/slurm/1_build_sidecar.sbatch"

step 3 "bucket annotations (array)" \
    sbatch --wait --export="$EXPORTS" --array="0-$((BUCKET_TASKS - 1))" \
        "$QDIR/slurm/2_bucket_annotations.sbatch"

step 4 "join annotations + build cubes" \
    sbatch --wait --export="$EXPORTS" "$QDIR/slurm/3_join_and_cube.sbatch"

# ============================================================ per ablation
step 5 "preview selection" \
    "$MQ" -m modalities quality preview \
        --selection "$SELECTION" --work_dir "$WORK"

step 6 "apply selection (filtered indexes)" \
    "$MQ" -m modalities quality apply \
        --selection "$SELECTION" \
        --registry "$REGISTRY" \
        --work_dir "$WORK" --output_dir "$WORK/$BLEND_NAME"

export_jsonl() {
    # One array task per dataset in the mix manifest. Each writes its own record; the
    # blend-wide manifest is merged afterwards, because concurrent tasks writing one shared
    # file would race and the last writer would erase the rest.
    n=$("$MQ" -c "import yaml,sys; m=yaml.safe_load(open('$WORK/mix/mix_manifest.yaml')); \
        print(len({d.get('source_dataset') or d['name'] for d in m['datasets']}))")
    if [[ "$n" -eq 0 ]]; then
        echo "no datasets in $WORK/mix/mix_manifest.yaml" >&2
        return 1
    fi
    echo "exporting $n dataset(s) as $n task(s)"
    sbatch --wait --export="$EXPORTS,MANIFEST=$WORK/mix/mix_manifest.yaml,OUT=$WORK/out" \
        --array="0-$((n - 1))" "$QDIR/slurm/4_export_jsonl.sbatch"
    "$MQ" -m modalities quality export-jsonl \
        --manifest "$WORK/mix/mix_manifest.yaml" \
        --registry "$REG" --output_dir "$WORK/out" --finalize_only
}

step 7 "export sampled documents as jsonl (array)" export_jsonl

echo
echo "Coverage per dataset: $WORK/join_report.json"
echo "Blend manifest:       $WORK/$BLEND_NAME/mix_manifest.yaml"
echo "Take the per-dataset 'ratio' values into a weighted_combined dataset in the training config."
