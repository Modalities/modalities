# Source this before running any stage by hand:
#
#   source config_files/data_preparation/quality/slurm/env.sh
#
# Every variable can be overridden by setting it first; this only fills in what is unset.
# Sourcing rather than exporting by hand is the point: a stage invoked with an empty $QDIR
# silently becomes "/slurm/...", and an empty $WORK turns "rm -rf $WORK/buckets" into
# "rm -rf /buckets". Both have happened.

export MQ="${MQ:-/data/user/richard.rutmann/venvs/modalities-quality/bin/python}"
export REPO="${REPO:-/home/richard.rutmann/repos/modalities}"
export QDIR="${QDIR:-$REPO/config_files/data_preparation/quality}"
export WORK="${WORK:-/data/user/richard.rutmann/annealing_blend}"
export HF_HOME="${HF_HOME:-/data/cache/hf_cache}"
export NUM_BUCKETS="${NUM_BUCKETS:-1024}"

if [[ -f "$HOME/.config/huggingface/token" ]]; then
    export HF_TOKEN="$(tr -d '\r\n' < "$HOME/.config/huggingface/token")"
fi

export EXPORTS="ALL,MQ=$MQ,QDIR=$QDIR,WORK=$WORK,NUM_BUCKETS=$NUM_BUCKETS,HF_HOME=$HF_HOME"

mkdir -p "$WORK" "$HOME/logs/quality"

# Fail loudly here rather than as a confusing error from sbatch or, worse, an rm against
# the wrong path.
for _v in MQ REPO QDIR WORK; do
    if [[ -z "${!_v:-}" ]]; then
        echo "quality env: $_v is empty -- refusing to continue" >&2
        return 1 2>/dev/null || exit 1
    fi
done
for _p in "$MQ" "$QDIR/annealing_registry.yaml" "$QDIR/slurm/2_bucket_annotations.sbatch"; do
    if [[ ! -e "$_p" ]]; then
        echo "quality env: expected to find $_p but it is missing" >&2
        return 1 2>/dev/null || exit 1
    fi
done
unset _v _p

echo "quality env ready:"
echo "  MQ    $MQ"
echo "  QDIR  $QDIR"
echo "  WORK  $WORK"
