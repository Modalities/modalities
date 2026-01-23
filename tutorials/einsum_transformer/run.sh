#!/bin/sh

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --rdzv-endpoint localhost:29804 --nnodes 1 --nproc_per_node 8 train.py