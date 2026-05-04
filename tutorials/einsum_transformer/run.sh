#!/bin/sh

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv-endpoint localhost:29804 --nnodes 1 --nproc_per_node 4 train.py