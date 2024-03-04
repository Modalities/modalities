# Supported Features
In this file, we list the already implemented, planned and in-progress features w.r.t. to improving downstream performance, throughput, multi-modality, and alignment. 

## Throughput Features

| Name                                  | Status           | Description                                                                                                       |
|---------------------------------------|------------------|-------------------------------------------------------------------------------------------------------------------|
| Mixed Precision Training              | supported        | Utilizes both single (FP32) and half precision (FP16) floating-point formats to speed up arithmetic computations while maintaining model accuracy. |
| Fully Sharded Data Parallel (FSDP)    | supported        | Optimizes distributed training by sharding the model parameters, gradients, and optimizer states across all GPUs, reducing memory overhead and enabling the training of larger models. |
| Gradient Accumulation                 | supported        | Allows for the use of larger batch sizes than what might fit in memory by accumulating gradients over multiple mini-batches before updating model weights. |
| CPU Offloading via FSDP               | supported        | Moves parts of the model or computation from GPU to CPU or other storage to manage GPU memory constraints. |
| Memmap for efficient data loading     | supported        | Optimizes the data pipeline to reduce I/O bottlenecks. |
| Activation Checkpointing              | supported        | Saves intermediate activations to memory only at certain points during the forward pass and recomputes them during the backward pass, reducing memory usage at the cost of additional computation. |
| Flash Attention                       | supported        | A highly optimized attention mechanism that significantly reduces the computational burden and memory footprint of attention calculations, enabling faster training and inference on large models. |


## Downstream Performance Features

## Multi-modality Features

## Alignment Features
