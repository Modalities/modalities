#!/bin/sh

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --rdzv-endpoint localhost:29501 --nnodes 1 --nproc_per_node 7 /path/to/Modalities/src/modalities/__main__.py run --config_file_path /path/to/Modalities/config_files/config.yaml