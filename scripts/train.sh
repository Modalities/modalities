#!/bin/sh

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --rdzv-endpoint localhost:29504 --nnodes 1 --nproc_per_node 6 $(which modalities) run --config_file_path ../config_files/config_example_mem_map_dataset.yaml