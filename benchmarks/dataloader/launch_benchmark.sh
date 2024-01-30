#!/bin/bash



INPUT_DIR="/tmp/i-do-not-exist.jsonl"


measure_modalities_preparation() {
    time (
        set -e
        test -f $INPUT_DIR
        rm -f ${INPUT_DIR/.jsonl/.idx}
        modalities create_memmap_index $INPUT_DIR &> /dev/null
        echo "finished memmap index creation"
        rm -f ${INPUT_DIR/.jsonl/.pbin}
        modalities create_packed_data $INPUT_DIR &> /dev/null
        echo "finished memmap packing"
    )
}


measure_modalities_initialization() {
  input_file=${INPUT_DIR/.jsonl/.pbin}
  python -m timeit -n 50 -r 5 -s "
import sys, io
null_device = io.StringIO()
from modalities.dataloader.dataset import PackedMemMapDatasetMegatron
from pathlib import Path
p = Path(\"${input_file}\")
  " -- "
sys.stdout = null_device  # deactivate stdout to avoid getting spammed
PackedMemMapDatasetMegatron(raw_data_path=p, block_size=1024, sample_key=\"sample\")
sys.stdout = sys.__stdout__  # reactivate stdout for timeit
"
}

measure_megatronLM_initialization() {
  input_file="${INPUT_DIR/.jsonl/.megLM.bin_text_document}"
  python -m timeit -n 50 -r 5 -s "
import sys, io
null_device = io.StringIO()
from modalities.dataloader.open_gptx_dataset.mmap_dataset import MMapIndexedDataset
p = \"${input_file}\"
  " -- "
sys.stdout = null_device  # deactivate stdout to avoid getting spammed
MMapIndexedDataset(p)
sys.stdout = sys.__stdout__  # reactivate stdout for timeit
"
}

measure_modalities_iteration() {
  input_file=${INPUT_DIR/.jsonl/.pbin}
  python -m timeit -n 5 -r 3 -s "
import random, sys, io
null_device = io.StringIO()
from modalities.dataloader.dataset import PackedMemMapDatasetMegatron
from pathlib import Path
p = Path(\"${input_file}\")
sys.stdout = null_device  # deactivate stdout to avoid getting spammed
dataset = PackedMemMapDatasetMegatron(raw_data_path=p, block_size=1024, sample_key=\"sample\")
random_indices = random.sample(range(len(dataset)), len(dataset))
sys.stdout = sys.__stdout__  # reactivate stdout for timeit
  " -- "
list(dataset)  # sequential access
for i in random_indices:
  dataset[i]
"
}


measure_megatronLM_iteration() {
  input_file="${INPUT_DIR/.jsonl/.megLM.bin_text_document}"
  python -m timeit -n 5 -r 3 -s "
import random, sys, io
null_device = io.StringIO()
from modalities.dataloader.open_gptx_dataset.mmap_dataset import MMapIndexedDataset
p = \"${input_file}\"
sys.stdout = null_device  # deactivate stdout to avoid getting spammed
dataset = MMapIndexedDataset(p)
random_indices = random.sample(range(len(dataset)), len(dataset))
sys.stdout = sys.__stdout__  # reactivate stdout for timeit
  " -- "
list(dataset)  # sequential access
for i in random_indices:
  dataset[i]
"
}


echo "MegatronLM:"
measure_megatronLM_iteration
echo "Modalities:"
measure_modalities_iteration