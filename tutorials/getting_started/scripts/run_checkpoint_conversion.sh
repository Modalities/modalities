#!/bin/sh
set -e 

# ---------------------------------------------
# bash run_checkpoint_conversion 
# ---------------------------------------------

#######################
### INPUT ARGUMENTS ###
#######################
if [ -z "$1" ] || [ -z "$2" ]  # if one of the two input arguments does not exist
  then
    echo "Need to specify arguments, e.g. bash run_checkpoint_conversion modalities_config output_dir"
    exit
fi

#############
### RUN #####
#############
echo "> run checkpoint conversion"
echo "python ../../src/modalities/conversion/gpt2/convert_gpt2.py" $1 $2 "--num_testruns 5"
python ../../src/modalities/conversion/gpt2/convert_gpt2.py $1 $2 --num_testruns 5
