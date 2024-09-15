# Warmstart Tutorial

In this tutorial, we demonstrate how you can continue the training from a checkpoint, e.g., after the training was interrupted or had crashed. 

## Prerequisites
We will use the data from the [Modalities in 15 mins Tutorial](../modalities_in_15_mins/modalities_demo.ipynb). 
If you haven't already, please run the data generation part of the notebook to generate the data.


# Running and warmstarting the model training

To train the model, we will execute the configuration file `pretrain.yaml` stored in folder `config`, as follows:

```bash
CUDA_VISIBLE_DEVICES="5,6" torchrun \
  --rdzv-endpoint localhost:29516  \
  --nnodes 1   \
  --nproc_per_node 2   \
  $(which modalities) run \
  --config_file_path configs/pre_training_config.yaml
```


We will interrupt the training manually (e.g., CTRL + C) after the 250 steps checkpoint has been written out to `data/checkpoints/`.

To continue the training from the checkpoint, we will execute the configuration file `warmstart.yaml` stored in folder `config`, running the command below. 
Note, that we have to change the paths under `warmstart_checkpoint_paths` in `warmstart.yaml` such that it points to the correct model and optimizer checkpoint files.

```bash
CUDA_VISIBLE_DEVICES="5,6" torchrun \
  --rdzv-endpoint localhost:29516  \
  --nnodes 1   \
  --nproc_per_node 2   \
  $(which modalities) run \
  --config_file_path configs/warmstart.yaml
```


Note that warmstarts do not require you to run the training on the exact same hardware. You can adapt the number of GPUs, number of tokens per batch, etc. in the command line arguments and in the configuration file. 
However, the training result is most likely not exactly the same as if you had continued the training on the same hardware.

We specify consistency checks in the configuration file, such as 
```yaml
  consistency_enforcement:
    enforce_tokens_per_step_conistency: true
    enforce_last_step_logged: false
    enforce_last_step_evaluated: false
    enforce_last_step_checkpointed: false
```
which can be relaxed to only print warnings instead of raising exceptions. 

