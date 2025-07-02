#!/bin/bash

if [ -z "$1" ] 
  then
    echo "Need to specify GPU devices as arguments, e.g. bash scripts/02_train_instruction_tuning_model.sh 0,1,2,3"
    exit
fi

if [ -z "$2" ] 
  then
    config_file_path=configs/train_instruct_model_fsdp1_config.yaml
else
    config_file_path=$2
fi

export CUDA_VISIBLE_DEVICES="$1"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c)
python -m torch.distributed.run \
  --nnodes 1 \
  --nproc_per_node $num_gpus \
  --rdzv-endpoint=0.0.0.0:29505 \
  $(which modalities) \
  run \
  --config_file_path $config_file_path
