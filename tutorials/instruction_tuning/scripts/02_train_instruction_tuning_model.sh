#!/bin/bash

export CUDA_VISIBLE_DEVICES="1,2,3,4"

python -m torch.distributed.run \
  --nnodes 1 \
  --nproc_per_node 4 \
  --rdzv-endpoint=0.0.0.0:29505 \
  src/modalities/__main__.py \
  run \
  --config_file_path tutorials/instruction_tuning/configs/train_instruct_model_fsdp1_config.yaml
