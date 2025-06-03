#!/bin/sh
#
# submit_experiments.sh
#
# Submits separate Slurm jobs for each desired node count.

# Accept node counts as arguments, otherwise use default.
if [ "$#" -gt 0 ]; then
    NODE_COUNTS="$@"
else
    NODE_COUNTS="1"  # Default values
fi

DATE_OF_RUN=$(date +"%Y-%m-%d__%H-%M-%S")

for NUM_NODES in $NODE_COUNTS
do
    echo "Submitting job with $NUM_NODES node(s)..."
    sbatch --nodes=$NUM_NODES \
           --job-name=15B_FSDP2_compiled_${NUM_NODES}nodes \
           --export=ALL,DATE_OF_RUN=$DATE_OF_RUN \
           submit_job.sbatch
done
