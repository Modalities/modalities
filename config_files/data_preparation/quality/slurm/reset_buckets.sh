#!/usr/bin/env bash
# Deletes the bucketed annotations so a split can be re-bucketed from scratch. A sharded
# bucketing run cannot clear the directory itself -- sibling tasks are writing into it --
# so clearing it is a deliberate separate step.
#
#   source .../env.sh && bash .../reset_buckets.sh
set -euo pipefail

# ":?" is the point of this script existing: an unset WORK would otherwise make this
# "rm -rf /buckets".
BUCKETS="${WORK:?WORK is not set -- source env.sh first}/buckets"

case "$BUCKETS" in
    /buckets|/|"") echo "refusing to delete $BUCKETS" >&2; exit 1 ;;
esac

if [[ ! -d "$BUCKETS" ]]; then
    echo "nothing to remove: $BUCKETS does not exist"
    exit 0
fi

echo "removing $BUCKETS"
echo "  split dirs: $(ls "$BUCKETS" | wc -l), parquet: $(find "$BUCKETS" -name '*.parquet' | wc -l)"
rm -rf "$BUCKETS"
echo "done. Sidecars under $WORK/sidecar are untouched."
