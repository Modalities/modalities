<p align="center">
  <img src="docs/source/banner.jpg">
</p>


<div align="center">
    <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue" alt="Python Versions">
    <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-green?logo=pytorch&logoColor=white" alt="PyTorch">
  </a>
  <a href="https://coveralls.io/github/Modalities/modalities">
    <img src="https://coveralls.io/repos/github/Modalities/modalities/badge.svg" alt="Coverage Status">
  </a>
 <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
  </a>
</div>




## Getting Started
For training and evaluation a model, feel free to checkout [this](https://github.com/Modalities/modalities/blob/main/examples/getting_started/README.md) getting started tutorial, in which we train a small, 60M-parameter GPT model on a tiny subset of the Redpajama V2 dataset. 
Also, see our Wiki and API reference documentation: https://modalities.github.io/modalities/

## Installation

There are two ways to install modalities. If you want to use the latest version, or if you want to modify the code base itself, you can install modalities directly from source. 

If you want to use modalities as a library and potentially register your custom components with modalities, you can install it directly via pip.


### Installation from source


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
Until they fix this, we have to run

```sh
pip install torch
```

Now modalities can be installed the repository via

```sh
cd modalities
pip install -e . 
```

If you want to contribute, have a look at `CONTRIBUTING.md`.

### Installation via pip

To install modalities via pip, run

```sh
pip install torch
pip install modalities
```

Note, that also here, torch has to be installed before installing modalities.


## Usage
For running the training endpoint on multiple GPUs run `CUDA_VISIBLE_DEVICES=2,3 torchrun --nnodes 1 --nproc_per_node 2 --rdzv-endpoint=0.0.0.0:29502 modalities run --config_file_path config_files/config.yaml`.

In the example above, we use `torchrun` to run the training endpoint on two GPUs. The `--nnodes` argument specifies the number of nodes in the cluster, `--nproc_per_node` specifies the number of processes per node, and `--rdzv-endpoint` specifies the rendezvous endpoint. The `modalities run` command specifies the training endpoint, and `--config_file_path` specifies the path to the configuration file.

Or, if you are a VsCode user, add this to your `launch.json`:
```json

        {
            "name": "Torchrun Main",
            "type": "python",
            "request": "launch",
            "module": "torch.distributed.run",
            "env": {
                "CUDA_VISIBLE_DEVICES": "0"
            },
            "args": [
                "--nnodes",
                "1",
                "--nproc_per_node",
                "2",
                "--rdzv-endpoint=0.0.0.0:29503",
                "src/modalities/__main__.py",
                "run",
                "--config_file_path",
                "config_files/config.yaml",
            ],
            "console": "integratedTerminal",
            "justMyCode": true,
            "envFile": "${workspaceFolder}/.env"
        }
```

## Supported Features
In the following, we list the already implemented, planned and in-progress features w.r.t. to improving downstream performance, throughput, multi-modality, and alignment. 

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

## Tutorials
Even though modalities significantly simplifies LLM training, there is still some technical complexity left. We provide a series of tutorials to help you get started with training and evaluating models using modalities.

- [Getting Started](examples/getting_started/README.md)</br>
  Brief overview on how to get started with modalities by training a small GPT model on a tiny subset of the Redpajama V2 dataset.

- [Library Usage](examples/library_usage/README.md)</br>
  How to use modalities as a library and register custom components with modalities.

- [Modalities in 15mins] </br>
  Jupyter notebook will be added soon

## Entry Points

We use [click](https://click.palletsprojects.com/en/) as a tool to add new entry points and their CLI arguments.
For this we have a main entry point from which all other entry points are started. 

The main entry point is `src/modalities/__main__.py:main()`. 
We register other sub-entrypoints by using our main `click.group`, called `main`, as follows: 
```python
@main.command(name="my_new_entry_point")
```

See the following full example:
```python
import click
import click_pathlib


@click.group()
def main() -> None:
    pass


config_option = click.option(
    "--config_file_path",
    type=click_pathlib.Path(exists=False),
    required=True,
    help="Path to a file with the YAML config file.",
)


@main.command(name="do_stuff")
@config_option
@click.option(
    "--my_cli_argument",
    type=int,
    required=True,
    help="New integer argument",
)
def entry_point_do_stuff(config_file_path: Path, my_cli_argument: int):
    print(f"Do stuff with {config_file_path} and {my_cli_argument}...)
    ...

if __name__ == "__main__":
    main()
```
With 
```toml
[project.scripts]
modalities = "modalities.__main__:main"
```
in our `pyproject.toml`, we can start only main with `modalities` (which does nothing), or a specific sub-entrypoint e.g. `modalities do_stuff --config_file_path config_files/config.yaml --my_cli_argument 3537`.

Alternatively, directly use `src/modalities/__main__.py do_stuff --config_file_path config_files/config.yaml --my_cli_argument 3537`.


## Contributing

Modalities welcomes your contributions! Please check out our
[contributing](CONTRIBUTING.md) guidelines regarding the details on formatting, testing,
etc.<br/><br/><br/>
Thanks so much to all of our amazing contributors!

<a href="https://github.com/modalities/modalities/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=modalities/modalities&r="  width="800px"/>
</a>

