#!/bin/sh
set -ex


# ---------------------------------------------
# sh pre_train_and_warmstart.sh 0 1
# (can only be run on 2 GPUs using this script)
# ---------------------------------------------

#######################
### INPUT ARGUMENTS ###
#######################
if [ -z "$1" ] || [ -z "$2" ]  # if one of the two input arguments does not exist
  then
    echo "Need to specify 2 GPU devices as arguments, e.g. sh pre_train_and_warmstart.sh 0 1"
    exit
fi
if [[ $1 =~ [^0-7] ]] || [[ $2 =~ [^0-7] ]]  # if one of the two input arguments is not an integer 0-7
    then
        echo "Need to specify integers 0-7 as arguments, e.g. sh pre_train_and_warmstart.sh 0 1"
        exit
fi

CUDA_VISIBLE_DEVICES="$1,$2"



echo "> run warmstart example on CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES

# cd to the directory of the script (absolute path)
cd "$(dirname "$0")"

rm -rf ../data/


# run preprocessing
modalities data create_raw_index --index_path ../data/mem_map/redpajama_v2_samples_512_train.idx ../../getting_started/data/raw/redpajama_v2_samples_512_train.jsonl
modalities data pack_encoded_data ../configs/tokenization_config_train.yaml

# run pretraining 

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --rdzv-endpoint localhost:29504 --nnodes 1 --nproc_per_node 2 $(which modalities) run --config_file_path ../configs/pre_training_config.yaml

# run warmstart
checkpoint_path=$(find ../data/checkpoints -name "last_checkpoint_info.json" -exec realpath {} \;)
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --rdzv-endpoint localhost:29504 --nnodes 1 --nproc_per_node 2 $(which modalities) warmstart --config_file_path ../configs/warmstart_config.yaml --last_checkpoint_info_file_path $checkpoint_path

# add some consistency checks
python check_checkpoint_consistency.py

echo "Finished warmstart example"