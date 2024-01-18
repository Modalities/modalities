#!/bin/sh

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --rdzv-endpoint localhost:29504 --nnodes 1 --nproc_per_node 8 $(which modalities) run --config_file_path /raid/s3/opengptx/max_lue/modalities/config_files/config_example_mem_map_dataset.yaml