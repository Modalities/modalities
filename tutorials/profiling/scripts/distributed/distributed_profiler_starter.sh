#!/usr/bin/env bash

cd "$(dirname "$0")"


# Environment
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


# Run torchrun
torchrun \
  --rdzv-endpoint localhost:29589 \
  --nnodes 1 \
  --nproc_per_node 8 \
  run_distributed_model_profiling.py
