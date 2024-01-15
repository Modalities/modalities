Quickstart
==========

**EDIT "docs/source/quickstart.rst" IN ORDER TO MAKE CHANGES HERE**

Installation
-----------
Setup a conda environment `conda create -n llm_gym python=3.10 & conda activate llm_gym` and install the requirements `pip install -e .`.

Setup Dataset
------------
To start a training you need to create memmap dataset out of a jsonl file first, then pack it, then run the training.

.. code-block:: bash

    # Create memmap dataset from jsonl file.
    llm_gym create_memmap_index <path/to/jsonl/file>

    # Create packed dataset.
    llm_gym create_packed_data <path/to/jsonl/file>

For example, using the lorem ipsum example:

.. code-block:: bash

    # Create memmap dataset from jsonl file.
    llm_gym create_memmap_index data/lorem_ipsum.jsonl

    # Create packed dataset.
    llm_gym create_packed_data data/lorem_ipsum.jsonl

Training
--------
To run a training environment variables in a multi-gpu setting are required.

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes 1 --nproc_per_node 2 --rdzv-endpoint=0.0.0.0:29502 src/llm_gym/__main__.py run --config_file_path config_files/config_lorem_ipsum.yaml

4. **Evaluation:**
   WIP add contents
