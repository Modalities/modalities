# Nemotron hybrid Mamba-Transformer models

Modalities supports hybrid Mamba-Transformer architectures with sparse mixture-of-experts feed-forward
layers, as introduced by NVIDIA's Nemotron-H and Nemotron-3 Nano families. The reference target is
**Nemotron-3 Nano 30B-A3B** ([arXiv:2512.20848](https://arxiv.org/abs/2512.20848)).

A ready-to-run pretraining configuration is at
[config_files/training/config_nemotron3_nano_30b_a3b_fsdp2.yaml](../../config_files/training/config_nemotron3_nano_30b_a3b_fsdp2.yaml).

## Architecture

Unlike a classical transformer, whose block bundles attention and a feed-forward network, a hybrid
model is a sequence of **single-operator pre-norm residual layers**:

```
x = x + operator(norm(x))
```

The stack is described by a **layer pattern** string with one character per layer:

| Symbol | Layer type |
|--------|------------|
| `M`    | Mamba-2 selective state space mixer |
| `*`    | Grouped-query causal self-attention |
| `E`    | Sparse mixture-of-experts feed-forward |
| `-`    | Dense squared-ReLU feed-forward |

Nemotron-3 Nano 30B-A3B uses 52 layers:

```
MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME
```

which is 23 Mamba-2, 23 MoE and just **6** attention layers. Because the Mamba-2 layers carry the
positional information, the model uses **no positional embeddings at all** (no RoPE, no learned
positions). The published hyperparameters are:

| Property | Value |
|----------|-------|
| Layers / model dimension | 52 / 2688 |
| Attention | 32 Q heads, 2 KV heads, head dim **128** (note: 32·128 ≠ 2688 by design) |
| Mamba-2 | 64 heads of dim 64 (inner dim 4096), state dim 128, 8 groups, conv kernel 4, chunk 128 |
| MoE | 128 routed experts of dim 1856, top-6, 2 shared experts (one fused MLP of 3712) |
| Router | sigmoid gating, expert bias for auxiliary-loss-free balancing, route scale 2.5, fp32 |
| Activation / norm | squared ReLU (non-gated), RMSNorm, no biases |
| Embeddings | untied, vocab 131072 |
| Parameters | 31.6B total / 3.2B active per token |

## Configuring the network components

The components of the network are registered as **layer specs**: declarative builders rather than
instantiated modules. Modalities' component factory memoises each config node, so injecting an
instantiated layer would make every layer of that type share one weight tensor. A spec is asked to
`build()` once per layer position, giving independent parameters per layer while keeping every
hyperparameter reachable from YAML.

```yaml
model_raw:
  component_key: model
  variant_key: nemotron
  config:
    n_embd: 2688
    n_layer: 52
    layer_pattern: "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
    layer_specs:
      "M":
        component_key: nemotron_layer_spec
        variant_key: mamba2
        config: {...}
      "E":
        component_key: nemotron_layer_spec
        variant_key: moe
        config: {...}
      "*":
        component_key: nemotron_layer_spec
        variant_key: attention
        config: {...}
```

Changing the Mamba/attention/MoE ratio is a one-line edit to `layer_pattern` (plus `n_layer`, which
is validated against it). Swapping an `E` for a `-` turns a sparse layer dense.

## Mamba-2 kernel backends

The Mamba-2 mixer has two interchangeable state space scan implementations, selected via
`ssd_backend`:

| Backend | Requirements | Use |
|---------|--------------|-----|
| `native` (default) | none | Pure PyTorch, runs on CPU and GPU, `torch.compile`-friendly. Correct but noticeably slower and more activation-memory hungry. |
| `fused` | `pip install -e '.[mamba]'`, CUDA | The Triton kernels from `mamba-ssm` / `causal-conv1d`. Use this for real training runs. |

The native backend implements the chunk-parallel block decomposition from the Mamba-2 paper and is
validated against a step-by-step transcription of the recurrence, so it is the reference the fused
path is checked against. A warning is emitted if `native` is used with `n_embd > 1024`.

## Mixture-of-experts load balancing

Nemotron combines two mechanisms, and Modalities exposes both:

1. **Auxiliary-loss-free balancing** (primary). Each router keeps an additive per-expert bias that
   affects expert *selection* only, never the output weights. An optimizer step pre-hook nudges the
   bias of under-loaded experts up and over-loaded experts down by a fixed step size (1e-3). Enable
   it by wrapping the optimizer:

   ```yaml
   optimizer:
     component_key: optimizer
     variant_key: moe_load_balanced
     config:
       expert_bias_update_rate: 1.0e-3
       model: {instance_key: initialized_model, pass_type: BY_REFERENCE}
       device_mesh: {instance_key: device_mesh, pass_type: BY_REFERENCE}
       optimizer:
         component_key: optimizer
         variant_key: adam_w
         config: {...}
   ```

   Attaching the update to the optimizer step (rather than the forward pass) is what makes it correct
   under gradient accumulation: token counts accumulate across micro-batches and are reduced across
   data-parallel ranks exactly once per step.

2. **The classic load-balancing loss** (secondary, coefficient 1e-4). Computed per sequence inside
   each MoE layer, summed by the model into the output dict under `aux_loss_key`, and added to the
   language modelling loss via the `weighted_sum` loss:

   ```yaml
   loss_fn:
     component_key: loss
     variant_key: weighted_sum
     config:
       weights: [1.0, 1.0]
       losses:
         - {component_key: loss, variant_key: clm_cross_entropy_loss, config: {...}}
         - {component_key: loss, variant_key: moe_aux_loss, config: {prediction_key: moe_aux_loss}}
   ```

## Weight initialization

The generic, regex-driven initializer (`model_type: nemotron`) handles all linear and embedding
weights. The state space parameters (`A_log`, `D`, `dt_bias`, `conv1d_*`) follow their own
distributions, defined in `Mamba2Mixer.reset_parameters` and matching the reference implementation;
the parameter-name filters deliberately exclude them so they cannot be overwritten. Recommended
weight decay exclusions:

```yaml
weight_decay_groups_excluded: [embedding, layernorm, ssm, router]
```

Decaying the SSM dynamics parameters or the router gate destabilizes training.

## Parallelism support

| Strategy | Status |
|----------|--------|
| FSDP2 (dp_shard / dp_replicate) | Supported. Expert weights shard along the expert dimension. |
| Activation checkpointing (all variants) | Supported via `layers_fqn: transformer.h`. |
| Pipeline parallelism | Supported via the `nemotron_stages_generator`, which balances stages by per-layer-type cost rather than layer count. |
| `torch.compile` per layer | Supported with the native backend. |
| Tensor parallelism | **Not supported yet.** Mamba's packed `in_proj` is a five-way unequal column split and the conv/`A_log`/`D`/`dt_bias` parameters need per-head sharding. |
| Expert parallelism | **Not supported.** The device mesh has no expert dimension. |
| Context parallelism | **Not supported yet.** Requires Mamba state passing across ranks. |

## Registered components

| Component type | Variant | Implementation |
|----------------|---------|----------------|
| `model` | `nemotron` | [NemotronModelFactory.get_nemotron_model](../../src/modalities/models/nemotron/nemotron_model_factory.py) |
| `nemotron_layer_spec` | `mamba2` | [Mamba2LayerSpec](../../src/modalities/models/nemotron/nemotron_layer_specs.py) |
| `nemotron_layer_spec` | `attention` | [NemotronAttentionLayerSpec](../../src/modalities/models/nemotron/nemotron_layer_specs.py) |
| `nemotron_layer_spec` | `moe` | [NemotronMoELayerSpec](../../src/modalities/models/nemotron/nemotron_layer_specs.py) |
| `nemotron_layer_spec` | `mlp` | [NemotronMLPLayerSpec](../../src/modalities/models/nemotron/nemotron_layer_specs.py) |
| `optimizer` | `moe_load_balanced` | [MoEBalancing.register_expert_bias_update_hook](../../src/modalities/models/components/moe/load_balancing.py) |
| `loss` | `moe_aux_loss` | [MoEAuxLoss](../../src/modalities/models/components/moe/moe_losses.py) |
| `loss` | `weighted_sum` | [WeightedSumLoss](../../src/modalities/models/components/moe/moe_losses.py) |
| `stages_generator` | `nemotron_stages_generator` | [NemotronStagesGenerator](../../src/modalities/models/nemotron/nemotron_stages_generator.py) |
| `mfu_calculator` | `nemotron` | [NemotronMFUCalculator](../../src/modalities/utils/nemotron_mfu.py) |
| `model_initialization` | `composed` (`model_type: nemotron`) | [ComposedInitializationRoutines](../../src/modalities/nn/model_initialization/composed_initialization.py) |
