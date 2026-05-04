## Einsum Transformer Tutorial

This tutorial showcases how new model architectures can used in modalities without requiring any code changes.
Particularly, we demonstrate this by training custom `EinsumTransformer` model. It uses a packed `.pbin` dataset and FSDP2 for sharded training.

**What’s inside**
- `train.py`: registers the custom model and launches the run.
- `einsum_transformer_config.yaml`: training + model config.
- `run.sh`: example `torchrun` command for 8 GPUs.
- `data/`: sample packed dataset (`.pbin` + `.idx`).

**Prerequisites**
- Modalities installed in your environment.
- A CUDA-capable, multi-GPU machine with PyTorch distributed (`torchrun`).
- Requires a prepared dataset to be passed `settings.paths.train_dataset_path` in `einsum_transformer_config.yaml`.  See the [Getting Started tutorial](../getting_started/README.md) on how to prepare datasets.

**Run (single node, 8 GPUs)**
```bash
cd /raid/s3/opengptx/max_lue/repositories/modalities/tutorials/einsum_transformer
./run.sh
```

**Notes**
- `num_workers` is set to `0` in the config to keep collator call counts deterministic.
- Outputs (checkpoints, logs, metrics) are written under `experiments/`.

**Customize**
- Model size and architecture: `model_raw` in `einsum_transformer_config.yaml`.
- Training schedule: `settings.step_profile` and `training_target` in the same file.
