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




## Getting Started
For training and evaluation a model, feel free to checkout [this](https://github.com/Modalities/modalities/blob/main/examples/getting_started/README.md) getting started tutorial, in which we train a small, 60M-parameter GPT model on a tiny subset of the Redpajama V2 dataset. 
Also, see our Wiki and API reference documentation: https://modalities.github.io/modalities/

## Installation

Create conda environment and activate it via 
```
conda create -n modalities python=3.10
conda activate modalities
```

then, install the repository via

```
pip install -e . 
```

If you want to contribute, have a look at `CONTRIBUTING.md`.



## Usage
For running the training endpoint on multiple GPUs run `CUDA_VISIBLE_DEVICES=2,3 torchrun --nnodes 1 --nproc_per_node 2 --rdzv-endpoint=0.0.0.0:29502 src/modalities/__main__.py run --config_file_path config_files/config.yaml`.

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
| Adaptive Batch Size Exploration       | planned         | Dynamically increases the training batch size during the training process to identify the maximum batch size that can be accommodated by a given GPU setup without causing memory overflow or performance degradation. |
| Node Failure Recovery                 | planned         | Implements mechanisms to automatically detect and recover from failures (e.g., node or GPU failures) in distributed training environments, ensuring that training can continue with minimal interruption even if one or more nodes / GPUs in the cluster fail. |



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

# Scaling Experiments

In the following, you can find the results of our scaling experiments performed on two HPC centers, namely [Leonardo Booster](https://leonardo-supercomputer.cineca.eu/hpc-system/) and [MareNostrum 5](https://www.bsc.es/ca/marenostrum/marenostrum-5). 

In a first step, we explored a **limited** set of different configurations (batch size, gradient accumulation steps, etc.) to get our baseline results. In a second step, we will focus on optimizing these configurations to maximize performance.


## Leonardo Booster  - NVIDIA A100 64GB
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

## MareNostrum5 - NVIDIA H100 64GB
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

