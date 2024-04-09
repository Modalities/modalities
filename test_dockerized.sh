#!/bin/bash
set -e 

docker build -t tmp/modalities .
# ensure you have atleast two gpus available
docker run --runtime nvidia tmp/modalities pytest
