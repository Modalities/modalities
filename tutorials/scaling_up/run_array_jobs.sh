#!/bin/sh
#
# submit_experiments.sh
#
# Submits separate Slurm jobs for each desired node count.

NODE_COUNTS="1 2" #  4 8 16 32 64"
DATE_OF_RUN=$(date +"%Y-%m-%d__%H-%M-%S")

for NUM_NODES in $NODE_COUNTS
do
    echo "Submitting job with $NUM_NODES node(s)..."
    sbatch --nodes=$NUM_NODES \
           --job-name=15B_FSDP2_compiled_${NUM_NODES}nodes \
           --export=ALL,DATE_OF_RUN=$DATE_OF_RUN \
           submit_job.sbatch
done
