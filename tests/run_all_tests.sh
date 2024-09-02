#!/bin/sh

#######################
### INPUT ARGUMENTS ###
#######################
if [ -z "$1" ] || [ -z "$2" ]  # if one of the two input arguments does not exist
  then
    echo "Need to specify 2 GPU devices as arguments, e.g. bash run_distributed_tests.sh 0 1"
    exit
fi
if [[ $1 =~ [^0-7] ]] || [[ $2 =~ [^0-7] ]]  # if one of the two input arguments is not an integer 0-7
    then
        echo "Need to specify integers 0-7 as arguments, e.g. bash run_distributed_tests.sh 0 1"
        exit 
fi

#################
### VARIABLES ###
#################
DEV0=$1 
DEV1=$2
COVERAGE=$3 # --cov or  --no-cov

#############
### TESTS ###
#############

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "$SCRIPT_DIR/.."


mkdir -p .coverage_reports
rm -f .coverage_reports/*

COVERAGE_FILE=.coverage_reports/.coverage.part0 python -m pytest tests/

sh tests/run_distributed_tests.sh $DEV0 $DEV1 --cov

# combine test coverage reports
cd .coverage_reports
coverage combine --keep
coverage report