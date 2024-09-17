# LoRA: Low-Rank Adaptation of Large Language Models

This directory contains an implementation of LoRA: Low-Rank Adaptation of Large Language Models. LoRA is a technique to
fine-tune large neural networks efficiently by restricting the updates during training to low-rank matrices.

## Introduction

LLMs have billions of parameters and require substantial computational resources for fine-tuning.
LoRA addresses this challenge by decomposing the weight update matrices into low-rank matrices, significantly reducing
the computational overhead while maintaining performance.

Basically, the weight matrix W is frozen during training. 
It is decomposed into matrices A and B with a lower rank than W.
These matrices store (among others) delta(W), i.e. the weight updates.
Then, W' = W + A*B. 
During evaluation, the weights are again merged for inference. 

<figure>
    <img alt="LoRA figure" src="/docs/source/lora_figure.png" width="30%" height="auto"/>
    <figcaption>
        Source: <a href="https://arxiv.org/abs/2106.09685">Paper</a>
    </figcaption>
</figure>

## Configuration

The pre-trained model is a HuggingFace [model](https://huggingface.co/HuggingFaceTB/SmolLM-1.7B).
It has been trained, but not instruction tuned.
An example configuration can be found [here](/config_files/training/config_lorem_ipsum_lora.yaml).
The configuration to load the HF model from a checkpoint is as follows:

```yaml
checkpointed_model:
  component_key: model
  variant_key: checkpointed
  config:
    checkpoint_loading:
      component_key: checkpoint_loading
      variant_key: torch
      config:
        device: 0
        precision: FP_32
    model:
      instance_key: huggingface_smol_llm_model
      pass_type: BY_REFERENCE
    checkpoint_path: .../SmolLM-1.7B_saved/model.bin
```

To cast the model into a lora_model, you need the following setting in the config file:

```yaml
lora_model:
  component_key: model
  variant_key: lora
  config:
    alpha: 1
    r: 2
    target_layers:
      - q_proj
      - k_proj
      - v_proj
      - o_proj
    model:
      instance_key: checkpointed_model
      pass_type: BY_REFERENCE
```

The config parameters needed are the following:

| Key           | Value              | Explanation                                             |
|---------------|--------------------|---------------------------------------------------------|
| lora_model    |                    | Name for your model.                                    |
| component_key | model              | Key for the component. Needs to be "model".             |
| variant_key   | lora               | Specifies the variant of the model, which is "lora".    |
| config        |                    | Configuration settings for the LoRA model.              |
| alpha         | 1                  | Scaling factor for the low-rank approximation.          |
| r             | 2                  | Rank of the low-rank approximation.                     |
| target_layers |                    | Names of the layers in the model where LoRA is applied. |
| model         |                    | Model you want to apply the LoRA conversion to.         |
| instance_key  | checkpointed_model | Key for the specific instance of the model              |
| pass_type     | BY_REFERENCE       | Indicates that the model is passed by reference         |

## Usage

Below is an example of how to use the LoRA implementation to fine-tune the loaded pre-trained language model on 3 GPUs.

```bash
python -m torch.distributed.run 
--nnodes 1 
--nproc_per_node 3 
--rdzv-endpoint=0.0.0.0:29555 
src/modalities/__main__.py run 
--config_file_path /config_files/training/config_lorem_ipsum_lora.yaml 
```
