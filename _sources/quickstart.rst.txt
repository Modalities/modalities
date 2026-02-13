Quickstart
====================================================

Installation
-----------------------------------------------------
Setup a conda environment `conda create -n modalities python=3.10 & conda activate modalities` and install the requirements `pip install -e .`.

Setup Dataset
-------------------------------------------------
To start a training you need to create memmap dataset out of a jsonl file first, then pack it, then run the training.

.. code-block:: bash

    # Create memmap dataset from jsonl file.
    modalities data create_raw_index <path/to/jsonl/file>

    # Create packed dataset.
    modalities data pack_encoded_data <path/to/jsonl/file>

For example, using the lorem ipsum example:

.. code-block:: bash

    # Create memmap dataset from jsonl file.
    modalities data create_raw_index data/lorem_ipsum.jsonl

    # Create packed dataset.
    modalities data pack_encoded_data data/lorem_ipsum.jsonl

Training
----------------------------------------------------
To run a training environment variables in a multi-gpu setting are required.

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes 1 --nproc_per_node 2 --rdzv-endpoint=0.0.0.0:29502 src/modalities/__main__.py run --config_file_path config_files/config_lorem_ipsum.yaml

4. **Evaluation:**
   WIP add contents
