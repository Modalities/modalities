<p align="center">
  <img src="docs/source/banner.jpg">
</p>


<div align="center">
    <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue" alt="Python Versions">
    <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-green?logo=pytorch&logoColor=white" alt="PyTorch">
  </a>
  <a href="https://coveralls.io/github/Modalities/modalities">
    <img src="https://coveralls.io/repos/github/Modalities/modalities/badge.svg?branch=main" alt="Coverage Status">
  </a>
 <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
  </a>
</div>

Modalities is a PyTorch-native framework for distributed training of Large Language Models (LLMs) and Foundation Models (FMs) at scale. Given the complexity of distributed training and rapid advancements in the field, we aim to provide a flexible and easy-to-use framework that enables researchers and practitioners to train and evaluate LLMs and FMs efficiently. Modalities is built on top of PyTorch and leverages the latest advancements in distributed training, such as Fully Sharded Data Parallel (FSDP), mixed precision training, Flash Attention and many more, to achieve state-of-the-art performance and throughput.

We successfully scaled Modalities up to 2048 GPUs on two HPC centers, namely [Leonardo Booster](https://leonardo-supercomputer.cineca.eu/hpc-system/) and [MareNostrum 5](https://www.bsc.es/ca/marenostrum/marenostrum-5), featuring Nvidia A100 and H100 GPUs, respectively. The results of our scaling experiments can be found [here](#scaling-experiments).

Besides its scalabilty, Modalities allows to seamlessly integrate new components and features, such as custom attention mechanisms, loss functions, optimizers or models. We provide a series of tutorials to help you get started with training and evaluating models using Modalities. We achieve this level of extensibility by having clear interfaces for each component type (e.g., model, optimizer, etc.), that a component must implement to be registered within Modalities at runtime. 

## Getting Started
For training and evaluation of a model, feel free to checkout [this](https://github.com/Modalities/modalities/blob/main/tutorials/getting_started/README.md) getting started tutorial, in which we train a small, 60M-parameter GPT model on a tiny subset of the Redpajama V2 dataset. 

## Installation

There are two ways to install Modalities. If you want to use the latest nightly version, or if you want to modify the code base itself, we recommend installing Modalities directly from source. 

If you want to use Modalities as a library and register your custom components with Modalities, you can install it directly via pip which provides you with the latest stable version.


### Option 1: Installation from source


Create a conda environment and activate it via 

```sh
conda create -n modalities python=3.10
conda activate modalities
```
Either clone the repository via
```sh
git clone git@github.com:Modalities/modalities.git
```
or download the repository as a zip file and extract it.
```
wget https://github.com/Modalities/modalities/archive/refs/heads/main.zip
unzip main.zip
```


Currently, the flash attention dependency cannot be installed without torch being installed beforehand.
Until the flash attention developers fix this, we have to run

```sh
pip install torch
```
beforehand.

Afterwards, Modalities can be installed via

```sh
cd modalities
pip install -e . 
```

### Option 2: Installation via pip

To install Modalities via pip, run

```sh
pip install torch
pip install modalities
```

Note, that also here, torch has to be installed before installing Modalities due to flash attention's dependency management.

## Usage
Modalities provides several entry points to interact with the framework. The following section lists the available entry points and their respective functionalities.


### Model Training

For model pretraining, we have to pass a configuration file that specifies the model architecture, optimizer, dataset, dataloader, and other training components. Additionally, we specify the number of nodes, the number of processes per node, and the rendezvous endpoint. 

Example:
```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv-endpoint localhost:29515 \
                                        --nnodes 1 \
                                        --nproc_per_node 4 \
                                        $(which modalities) run --config_file_path configs/pretraining_config.yaml
```

Explanation:

* `CUDA_VISIBLE_DEVICES=0,1,2,3`: This environment variable specifies which GPUs will be used for the job. In the example above, the four GPUs with IDs 0, 1, 2, 3 are selected for training.

* `torchrun`: This is a utility from PyTorch used to launch distributed training. It automatically manages multiple processes for distributed training.

* `--rdzv-endpoint localhost:29515`: Specifies the rendezvous endpoint. Here, localhost is the machine's address, and 29515 is the port. The rendezvous endpoint coordinates the processes involved in distributed training.

* `--nnodes 1`: Specifies the number of nodes to be used in the distributed setup. In the example above, a single-node setup is used.

* `--nproc_per_node 4`: This argument tells torchrun how many processes to launch on each node. In the example above, 4 processes are launched per node, corresponding to the 4 GPUs (IDs 0, 1, 2, 3) specified by CUDA_VISIBLE_DEVICES.

* `$(which modalities) run`: This part dynamically finds the path to the Modalities executable and runs it. The run command triggers the main process to start the training.

* `--config_file_path configs/pretraining_config.yaml`: The --config_file_path argument provides the path to the configuration file for the training job. In the example above, it is given by `configs/pretraining_config.yaml`. A configuraton file contains an exhaustive parameterization for all the training components (e.g., dataset, model, optimizer, etc.), making training fully reproducible. An example configuration file can be found [here](tutorials/getting_started/example_config.yaml), and a complete list of components available in Modalities is provided [here](docs/components/components.md).

If you are a VSCode user, you may want to add this to your `launch.json`:
```json

        {
            "name": "Torchrun Main",
            "type": "python",
            "request": "launch",
            "module": "torch.distributed.run",
            "env": {
                "CUDA_VISIBLE_DEVICES": "0,1,2,3"
            },
            "args": [
                "--nnodes",
                "1",
                "--nproc_per_node",
                "4",
                "--rdzv-endpoint=0.0.0.0:29515",
                "src/modalities/__main__.py",
                "run",
                "--config_file_path",
                "config_files/pretraining_config.yaml",
            ],
            "console": "integratedTerminal",
            "justMyCode": true,
            "envFile": "${workspaceFolder}/.env"
        }
```
It will allow you to run the training endpoint directly from VSCode and debug it.

### Raw Training Dataset Indexation

The goal of the indexation process is to determine the starting byte position and length of each document in the raw data file. Subsequently, the index file is used to efficiently access the raw data during tokenization.

Example:
```sh
modalities data create_raw_index --index_path data/preprocessed/fineweb_edu_num_docs_483606.idx \
                                               data/raw/fineweb_edu_num_docs_483606.jsonl
```

Explanation:

The `modalities data create_raw_index` command triggers the process of creating the index from the raw data. The `--index_path` argument specifies the location where the generated index file will be saved. In this example, the index will be stored at `data/preprocessed/fineweb_edu_num_docs_483606.idx`. The last part, i.e., `data/raw/fineweb_edu_num_docs_483606.jsonl` is the input file in JSONL (JSON Lines) format containing the raw data. The command will process this file to create the index.

### Raw Training Dataset Tokenization

Tokenization is the process of converting raw text data into a sequence of tokens that can be used as input to the model. The tokenization requires a configuration file, fully describing the tokenization process, making it fully reproducible. An example tokenization config can be found [here](tutorials/getting_started/example_dataset_config_train.yaml).

Example:
```sh
modalities data pack_encoded_data configs/tokenization_config.yaml
```

### Inference

For inference on a model checkpoint, we have to pass a configuration file that specifies the full inference setup. An example inference config can be found [here](tutorials/getting_started/example_text_generation_config.yaml).

Example:

```sh
modalities generate_text --config_file_path example_text_generation_config.yaml 

```

## Tutorials
Even though Modalities significantly simplifies LLM training, there is still some technical complexity left. We provide a series of tutorials to help you get started with training and evaluating models using Modalities.

- [Modalities in 15mins](tutorials/modalities_in_15_mins/README.md) </br>
  Train a dense model with Modalities in 15 minutes

- [Getting Started](tutorials/getting_started/README.md)</br>
  Brief overview on how to get started with Modalities by training a small GPT model on a tiny subset of the Redpajama V2 dataset.

- [Warmstart](tutorials/warmstart/README.md) </br>
  Continue the training from a checkpoint, e.g., after the training was interrupted or had crashed.

- [Library Usage](tutorials/library_usage/README.md)</br>
  How to use Modalities as a library and register custom components with Modalities.




## Supported Features
In the following, we list the most important features of Modalities.

### Throughput Features

| Name                                  | Status           | Description                                                                                                       |
|---------------------------------------|------------------|-------------------------------------------------------------------------------------------------------------------|
| Mixed Precision Training              | supported        | Utilizes both single (FP32) and half precision (FP16) floating-point formats to speed up arithmetic computations while maintaining model accuracy. Support for bf16|
| Fully Sharded Data Parallel (FSDP)    | supported        | Optimizes distributed training by sharding the model parameters, gradients, and optimizer states across all GPUs, reducing memory overhead and enabling the training of larger models. |
| Gradient Accumulation                 | supported        | Allows for the use of larger batch sizes than what might fit in memory by accumulating gradients over multiple mini-batches before updating model weights. |
| CPU Offloading via FSDP               | supported        | Moves parts of the model or computation from GPU to CPU or other storage to manage GPU memory constraints. |
| Memmap for efficient data loading     | supported        | Optimizes the data pipeline to reduce I/O bottlenecks. |
| Activation Checkpointing              | supported        | Saves intermediate activations to memory only at certain points during the forward pass and recomputes them during the backward pass, reducing memory usage at the cost of additional computation. |
| Flash Attention                       | supported        | A highly optimized attention mechanism that significantly reduces the computational burden and memory footprint of attention calculations, enabling faster training and inference on large models. |
| Tensor Parallelism                    | prototype       | Implementing vertical model sharding, as an efficient model parallelism technique|
| Sequence Parallelism                  | prototype       | Variant of Tensor Parallelism that shard on the sequence dimension |
| FSDP 2                                | prototype       | Improved version of the original FSDP |
| Torch Compile                         | prototype       | Speeds up tensor operations by JIT compiling tensor operations into optimized kernels |
| Deferred Initialisation               | prototype       | Instead of instantiating the model in CPU RAM, the modules are instantiated as fake tensors and operations are recorded. Once sharded (e.g., via FSDP), each rank only instantiates the local tensors by replaying the tensor operations.|
| Adaptive Batch Size Exploration       | planned         | Dynamically increases the training batch size during the training process to identify the maximum batch size that can be accommodated by a given GPU setup without causing memory overflow or performance degradation. |
| Node Failure Recovery                 | planned         | Implements mechanisms to automatically detect and recover from failures (e.g., node or GPU failures) in distributed training environments, ensuring that training can continue with minimal interruption even if one or more nodes / GPUs in the cluster fail. |
| Loss Parallelism                      | planned       | Reduces memory footprint and communication overhead by computing the loss locally on each rank. |


### Downstream Performance Features

| Name                           | Status           | Description                                                                                                       |
|--------------------------------|------------------|-------------------------------------------------------------------------------------------------------------------|
| SwiGLU                         | supported         | A nonlinear activation function combining Gated Linear Units (GLU) with Swish for enhancing model capacity and learning efficiency. |
| Weight Decay                   | supported        | Regularization technique that adds a penalty on the size of weights, encouraging smaller weights to reduce overfitting and improve generalization. |
| Weight Initialization          | supported        | Choose between different, configurable weight initialization techniques to stabilize training. |
| RMSNorm (pre-normalization)    | supported        | Normalizes the pre-activation weights in a layer to stabilize training, often used as an alternative to LayerNorm for improved training dynamics. |
| Rotary Positional Embeddings (RoPE) | supported  | Encodes sequence position information into attention mechanisms, preserving relative positional information and improving model's understanding of sequence order. |
| Grouped-query Attention (GQA)  | supported    | Enhances attention mechanisms by grouping queries to reduce computation and memory footprint while maintaining or improving performance. |
| Learning Rate Scheduler        | supported     | Adjusts the learning rate during training according to a predefined schedule (e.g., step decay, exponential decay) to improve convergence and performance. |
| Gradient Clipping              | supported         | Prevents exploding gradients by clipping the gradients of an optimization algorithm to a maximum value, thereby stabilizing training. |
| Training Warmup                | supported          | Gradually increases the learning rate from a low to a high value during the initial phase of training to stabilize optimization. |
| Loss Masking                   | planned          | Ignores or gives less weight to certain data points in the loss function, often used in tasks with variable-length sequences to ignore padding tokens or in more specific usecases such as GAtt. |
| Knowledge Distillation         | planned  | Transfers knowledge from a larger, complex model to a smaller, more efficient model, improving the smaller model's performance without the computational cost of the larger model.|
| Hyperparameter Optimization    | planned          | Grid search for various hyperparameter such as LR, Optimizer arguments etc. Also the integration of µP might be interesting |


## Scaling Experiments

In the following, you can find the results of our scaling experiments performed on two HPC centers, namely [Leonardo Booster](https://leonardo-supercomputer.cineca.eu/hpc-system/) and [MareNostrum 5](https://www.bsc.es/ca/marenostrum/marenostrum-5). 

In a first step, we explored a **limited** set of different configurations (batch size, gradient accumulation steps, etc.) to get our baseline results. In a second step, we will focus on optimizing these configurations to maximize throughput.


### Leonardo Booster  - NVIDIA A100 64GB
|  # Params (B) | #GPUs | Samples/s | GradAccm | MBS | GBS | Sequence Length | Precision | Sharding | AC | GPU Type | MFU |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2.7 | 8 | 18.63 | 1 | 2 | 16 | 4096 | BF_16 | FULL_SHARD | False |  A100  | 0.5847 |
| 2.7 | 8 | 18.43 | 1 | 2 | 16 | 4096 | BF_16 | HYBRID_SHARD | False |  A100  | 0.5786 |
| 2.7 | 16 | 36.68 | 1 | 2 | 32 | 4096 | BF_16 | HYBRID_SHARD | False |  A100  | 0.5757 |
| 2.7 | 16 | 36.96 | 1 | 2 | 32 | 4096 | BF_16 | FULL_SHARD | False |  A100  | 0.58 |
| 2.7 | 32 | 72.63 | 1 | 2 | 64 | 4096 | BF_16 | HYBRID_SHARD | False |  A100  | 0.5699 |
| 2.7 | 32 | 73.76 | 1 | 2 | 64 | 4096 | BF_16 | FULL_SHARD | False |  A100  | 0.5788 |
| 2.7 | 64 | 146.12 | 1 | 2 | 128 | 4096 | BF_16 | FULL_SHARD | False |  A100  | 0.5733 |
| 2.7 | 64 | 145.31 | 1 | 2 | 128 | 4096 | BF_16 | HYBRID_SHARD | False |  A100  | 0.5701 |
| 2.7 | 128 | 285.64 | 1 | 2 | 256 | 4096 | BF_16 | HYBRID_SHARD | False |  A100  | 0.5603 |
| 2.7 | 128 | 205.96 | 1 | 2 | 256 | 4096 | BF_16 | FULL_SHARD | False |  A100  | 0.404 |
| 2.7 | 256 | 495.44 | 1 | 2 | 512 | 4096 | BF_16 | HYBRID_SHARD | False |  A100  | 0.4859 |
| 2.7 | 256 | 303.17 | 1 | 2 | 512 | 4096 | BF_16 | FULL_SHARD | False |  A100  | 0.2974 |
| 2.7 | 8 | 19.94 | 1 | 4 | 32 | 4096 | BF_16 | FULL_SHARD | False |  A100 | 0.626 |
| 2.7 | 16 | 39.68 | 1 | 4 | 64 | 4096 | BF_16 | FULL_SHARD | False |  A100 | 0.6227 |
| 2.7 | 32 | 78.3 | 1 | 4 | 128 | 4096 | BF_16 | FULL_SHARD | False |  A100 | 0.6144 |
| 2.7 | 64 | 155.21 | 1 | 4 | 256 | 4096 | BF_16 | FULL_SHARD | False |  A100 | 0.6089 |
| 2.7 | 128 | 303.76 | 1 | 4 | 512 | 4096 | BF_16 | FULL_SHARD | False |  A100 | 0.5959 |
| 2.7 | 256 | 506.08 | 1 | 4 | 1024 | 4096 | BF_16 | FULL_SHARD | False |  A100 | 0.4964 |
| 6.7 | 8 | 9.28 | 1 | 2 | 16 | 4096 | BF_16 | FULL_SHARD | False |  A100  | 0.6867 |
| 6.7 | 16 | 18.35 | 1 | 2 | 32 | 4096 | BF_16 | FULL_SHARD | False |  A100  | 0.6789 |
| 6.7 | 32 | 36.65 | 1 | 2 | 64 | 4096 | BF_16 | FULL_SHARD | False |  A100  | 0.6782 |
| 6.7 | 64 | 72.72 | 1 | 2 | 128 | 4096 | BF_16 | FULL_SHARD | False |  A100  | 0.6727 |
| 6.7 | 128 | 131.59 | 1 | 2 | 256 | 4096 | BF_16 | FULL_SHARD | False |  A100  | 0.6086 |
| 6.7 | 256 | 225.24 | 1 | 2 | 512 | 4096 | BF_16 | FULL_SHARD | False |  A100  | 0.5209 |

Further scaling results can be found at [Leonardo Booster Scaling Experiments](https://github.com/Modalities/modalities/blob/scaling_experiments/docs/scaling_experiments/scaling_leonardo.md)

### MareNostrum 5 - NVIDIA H100 64GB
|  # Params (B) | #GPUs | Samples/s | GradAccm | MBS | GBS | Sequence Length | Precision | Sharding | AC | GPU Type | MFU |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2.7 | 4 | 15.06 | 1 | 2 | 8 | 4096 | BF_16 | FULL_SHARD | False |  H100 | 0.2983 |
| 2.7 | 4 | 15.14 | 1 | 2 | 8 | 4096 | BF_16 | HYBRID_SHARD | False |  H100 | 0.2998 |
| 2.7 | 8 | 29.6 | 1 | 2 | 16 | 4096 | BF_16 | HYBRID_SHARD | False |  H100 | 0.2931 |
| 2.7 | 8 | 29.75 | 1 | 2 | 16 | 4096 | BF_16 | FULL_SHARD | False |  H100 | 0.2946 |
| 2.7 | 16 | 58.7 | 1 | 2 | 32 | 4096 | BF_16 | HYBRID_SHARD | False |  H100 | 0.2906 |
| 2.7 | 16 | 59.61 | 1 | 2 | 32 | 4096 | BF_16 | FULL_SHARD | False |  H100 | 0.2951 |
| 2.7 | 32 | 117.07 | 1 | 2 | 64 | 4096 | BF_16 | HYBRID_SHARD | False |  H100 | 0.2898 |
| 2.7 | 32 | 117.62 | 1 | 2 | 64 | 4096 | BF_16 | FULL_SHARD | False |  H100 | 0.2912 |
| 2.7 | 64 | 235.96 | 1 | 2 | 128 | 4096 | BF_16 | FULL_SHARD | False |  H100 | 0.292 |
| 2.7 | 64 | 234.65 | 1 | 2 | 128 | 4096 | BF_16 | HYBRID_SHARD | False |  H100 | 0.2904 |
| 2.7 | 128 | 455.87 | 1 | 2 | 256 | 4096 | BF_16 | FULL_SHARD | False |  H100 | 0.2821 |
| 2.7 | 256 | 883.07 | 1 | 2 | 512 | 4096 | BF_16 | FULL_SHARD | False |  H100 | 0.2732 |
| 2.7 | 512 | 1831.71 | 1 | 2 | 1024 | 4096 | BF_16 | HYBRID_SHARD | False |  H100 | 0.2834 |
| 2.7 | 512 | 1365.31 | 1 | 2 | 1024 | 4096 | BF_16 | FULL_SHARD | False |  H100 | 0.2112 |
| 2.7 | 1024 | 1105.99 | 1 | 2 | 2048 | 8192 | BF_16 | FULL_SHARD | False |  H100 | 0.2071 |
| 2.7 | 1024 | 3618.0 | 1 | 2 | 2048 | 4096 | BF_16 | HYBRID_SHARD | False |  H100 | 0.2799 |
| 28 | 16 | 2.9 | 1 | 1 | 16 | 8192 | BF_16 | FULL_SHARD | True |  H100  | 0.2998 |
| 28 | 32 | 5.53 | 1 | 1 | 32 | 8192 | BF_16 | FULL_SHARD | True |  H100  | 0.2863 |
| 28 | 64 | 11.61 | 1 | 1 | 64 | 8192 | BF_16 | FULL_SHARD | True |  H100  | 0.3003 |
| 28 | 128 | 22.95 | 1 | 1 | 128 | 8192 | BF_16 | FULL_SHARD | True |  H100  | 0.2968 |
| 28 | 256 | 44.22 | 1 | 1 | 256 | 8192 | BF_16 | FULL_SHARD | True |  H100  | 0.286 |
| 28 | 512 | 87.36 | 1 | 1 | 512 | 8192 | BF_16 | FULL_SHARD | True |  H100  | 0.2825 |
| 28 | 512 | 87.56 | 1 | 1 | 512 | 8192 | BF_16 | FULL_SHARD | True |  H100  | 0.2831 |
| 28 | 1024 | 162.16 | 1 | 1 | 1024 | 8192 | BF_16 | FULL_SHARD | True |  H100  | 0.2622 |
| 28 | 2048 | 297.0 | 1 | 1 | 2048 | 8192 | BF_16 | FULL_SHARD | True |  H100  | 0.2401 |

Further scaling results can be found at [MareNostrum5 Scaling Experiments](https://github.com/Modalities/modalities/blob/scaling_experiments/docs/scaling_experiments/scaling_mn5.md)

![Scaling Plot for a 28B model with a sequence length of 8192 tokens](https://github.com/Modalities/modalities/blob/scaling_experiments/docs/scaling_experiments/scaling_28B_mbs_1_ac_True.png)


## Contributing

Modalities welcomes your contributions! Please check out our
[contributing](CONTRIBUTING.md) guidelines regarding the details on formatting, testing,
etc.<br/><br/><br/>
Thanks so much to all of our amazing contributors!

<a href="https://github.com/modalities/modalities/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=modalities/modalities&r="  width="800px"/>
</a>

