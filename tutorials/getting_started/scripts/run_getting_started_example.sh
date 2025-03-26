#!/bin/sh
set -e 

# ---------------------------------------------
# bash run_getting_started_example.sh 0 1
# (can only be run on 2 GPUs using this script)
# ---------------------------------------------

#######################
### INPUT ARGUMENTS ###
#######################
if [ -z "$1" ] || [ -z "$2" ]  # if one of the two input arguments does not exist
  then
    echo "Need to specify 2 GPU devices as arguments, e.g. bash run_getting_started_example.sh 0 1"
    exit
fi
if [[ $1 =~ [^0-7] ]] || [[ $2 =~ [^0-7] ]]  # if one of the two input arguments is not an integer 0-7
    then
        echo "Need to specify integers 0-7 as arguments, e.g. bash run_getting_started_example.sh 0 1"
        exit
fi

CUDA_VISIBLE_DEVICES="$1,$2"

#############
### RUN #####
#############
echo "> run getting_started_examples on CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES

modalities data create_raw_index --index_path data/mem_map/redpajama_v2_samples_512_train.idx data/raw/redpajama_v2_samples_512_train.jsonl
modalities data create_raw_index --index_path data/mem_map/redpajama_v2_samples_512_test.idx data/raw/redpajama_v2_samples_512_test.jsonl
modalities data pack_encoded_data configs/example_dataset_config_train.yaml
modalities data pack_encoded_data configs/example_dataset_config_test.yaml
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --rdzv-endpoint localhost:29505 --nnodes 1 --nproc_per_node 2 $(which modalities) run --config_file_path configs/example_config.yaml
