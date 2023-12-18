#!/bin/sh

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --rdzv-endpoint localhost:29501 --nnodes 1 --nproc_per_node 7 /raid/s3/opengptx/max_lue/LLMgym/src/llm_gym/__main__.py run --config_file_path /raid/s3/opengptx/max_lue/LLMgym/config_files/config.yaml