#!/bin/sh

# ---------------------------------------------------
# bash run_getting_started_example.sh 0,1,2,3,4,5,6,7
# ---------------------------------------------------

#######################
### INPUT ARGUMENTS ###
#######################
if [ -z "$1" ]  # if input argument does not exist
  then
    echo "Need to specify the GPU devices as arguments, e.g. bash run_getting_started_example.sh 0,1,2,3,4,5,6,7"
    exit
fi
CUDA_VISIBLE_DEVICES=$1

first_character=${1:0:1}
if [[ $first_character =~ [^0-7] ]]   # if the first character is not an integer 0-7
    then
        echo "First character of specified argument needs to be an integer, e.g. bash run_getting_started_example.sh 0,1,2,3,4,5,6,7"
        exit
fi

last_character=${1:0-1}
if [[ $last_character =~ [^0-7] ]]   # if the first character is not an integer 0-7
    then
        echo "Last character of specified argument needs to be an integer, e.g. bash run_getting_started_example.sh 0,1,2,3,4,5,6,7"
        exit
fi

#############
### RUN #####
#############
echo "> run getting_started_examples on CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES

modalities data create_raw_index --index_path data/mem_map/redpajama_v2_samples_512_train.idx data/raw/redpajama_v2_samples_512_train.jsonl
modalities data create_raw_index --index_path data/mem_map/redpajama_v2_samples_512_test.idx data/raw/redpajama_v2_samples_512_test.jsonl
modalities data pack_encoded_data example_dataset_config_train.yaml
modalities data pack_encoded_data example_dataset_config_test.yaml
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --rdzv-endpoint localhost:29505 --nnodes 1 --nproc_per_node 2 $(which modalities) run --config_file_path example_config.yaml
