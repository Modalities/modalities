#!/bin/sh

CUDA_VISIBLE_DEVICES=0,1 torchrun --rdzv-endpoint localhost:29504 --nnodes 1 --nproc_per_node 2 main.py