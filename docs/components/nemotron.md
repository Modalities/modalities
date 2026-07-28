# Nemotron hybrid Mamba-Transformer models

Modalities supports hybrid Mamba-Transformer architectures with sparse mixture-of-experts feed-forward
layers, as introduced by NVIDIA's Nemotron-H and Nemotron-3 Nano families. The reference target is
**Nemotron-3 Nano 30B-A3B** ([arXiv:2512.20848](https://arxiv.org/abs/2512.20848)).

Two ready-to-run configurations are provided:

| Config | Scale | Verified on |
|--------|-------|-------------|
| [config_nemotron3_nano_30b_a3b_fsdp2.yaml](../../config_files/training/config_nemotron3_nano_30b_a3b_fsdp2.yaml) | Full 52-layer 30B-A3B | Not run (needs tensor/expert parallelism for real throughput) |
| [config_lorem_ipsum_nemotron_nano_fsdp2.yaml](../../config_files/training/config_lorem_ipsum_nemotron_nano_fsdp2.yaml) | Full width, 16 layers, 9.67B params | 4x A100-SXM4-80GB, 162 steps, 46.1 GiB peak per GPU |

The lorem-ipsum config keeps every width hyperparameter of the real model (dimension 2688, 128
experts, 64 Mamba heads, 32/2 attention heads) and only trims the depth, so it exercises the real
architecture end to end. Shorten `layer_pattern` to `"MEMEM*EME"` with `n_layer: 9` for 40GB cards
(5.64B parameters, 28.9 GiB peak per GPU).

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

## Attribution

Modalities is MIT licensed. The files below adapt code from third-party projects and carry the
corresponding notices in their file headers, as required by those licenses.

**NVIDIA Megatron-LM** — Copyright (c) NVIDIA CORPORATION, Apache License 2.0.
Portions additionally Copyright (c) 2024, Tri Dao, Albert Gu.

| File | What was adapted | Upstream |
|------|------------------|----------|
| [mamba2_mixer.py](../../src/modalities/models/components/mamba2/mamba2_mixer.py) | Packed `[z, x, B, C, dt]` projection layout; `conv1d`/`A_log`/`D`/`dt_bias` shapes and init distributions | `megatron/core/ssm/mamba_mixer.py` |
| [layer_pattern.py](../../src/modalities/models/nemotron/layer_pattern.py) | Layer pattern symbols (`M`, `*`, `E`, `-`) | `megatron/core/models/hybrid/hybrid_layer_allocation.py::Symbols` |
| [nemotron_layers.py](../../src/modalities/models/nemotron/nemotron_layers.py) | Single-operator pre-norm residual layer structure | `megatron/core/ssm/mamba_layer.py`, `.../hybrid/hybrid_block.py` |
| [nemotron_layer_specs.py](../../src/modalities/models/nemotron/nemotron_layer_specs.py) | Declarative layer-spec/builder pattern | `megatron/core/transformer/spec_utils.py::ModuleSpec`, `.../hybrid/hybrid_layer_specs.py` |
| [nemotron_model.py](../../src/modalities/models/nemotron/nemotron_model.py) | Hybrid model structure (pattern-driven stack) | `megatron/core/models/hybrid/hybrid_model.py` |
| [router.py](../../src/modalities/models/components/moe/router.py) | Sigmoid scoring, selection-only expert bias, top-k renormalization | `megatron/core/transformer/moe/moe_utils.py::topk_routing_with_score_function` |
| [moe.py](../../src/modalities/models/components/moe/moe.py) | Load-balancing loss formula and its sequence-level variant | `.../moe_utils.py::switch_load_balancing_loss_func` |
| [load_balancing.py](../../src/modalities/models/components/moe/load_balancing.py) | Sign-based expert bias update rule | `.../moe_utils.py::get_updated_expert_bias` |
| [experts.py](../../src/modalities/models/components/moe/experts.py), [nemotron_mlp.py](../../src/modalities/models/nemotron/nemotron_mlp.py) | Squared ReLU activation | `megatron/core/activations.py` |

**state-spaces/mamba** — Copyright (c) 2024, Tri Dao, Albert Gu, Apache License 2.0.

| File | What was adapted | Upstream |
|------|------------------|----------|
| [ssd.py](../../src/modalities/models/components/mamba2/ssd.py) | Chunk-parallel SSD block decomposition (`ssd_minimal_discrete` / `segsum`); `GatedRMSNorm` semantics | `ssd_minimal_discrete`, `mamba_ssm.ops.triton.layernorm_gated.rmsnorm_fn` |

**Meta TorchTitan** — BSD 3-Clause License. (Modalities already adapts from TorchTitan elsewhere.)

| File | What was adapted | Upstream |
|------|------------------|----------|
| [experts.py](../../src/modalities/models/components/moe/experts.py) | Stacked per-expert weight layout; `torch._grouped_mm` over expert-sorted tokens | `torchtitan/models/common/moe.py::GroupedExperts` |
| [moe.py](../../src/modalities/models/components/moe/moe.py) | Dispatch/combine structure; expert-bias and token-count buffers | `.../moe.py::MoE` |
| [router.py](../../src/modalities/models/components/moe/router.py) | Router interface shape | `.../moe.py::TokenChoiceTopKRouter` |
| [load_balancing.py](../../src/modalities/models/components/moe/load_balancing.py) | Applying the bias update as an optimizer step pre-hook | `.../moe.py` |
| [nemotron_stages_generator.py](../../src/modalities/models/nemotron/nemotron_stages_generator.py) | Pipeline split-point structure (via Modalities' own `StagesGenerator`) | TorchTitan pipeline utilities |

**Meta Llama** — [facebookresearch/llama](https://github.com/facebookresearch/llama).

| File | What was adapted | Upstream |
|------|------------------|----------|
| [nemotron_attention.py](../../src/modalities/models/nemotron/nemotron_attention.py) | Grouped-query key/value head repetition (`_repeat_kv`), via Modalities' own GPT-2 | `llama/model.py` |

**NVIDIA Megatron-Bridge** — Copyright (c) 2026, NVIDIA CORPORATION, Apache License 2.0.
The hyperparameter values in
[config_nemotron3_nano_30b_a3b_fsdp2.yaml](../../config_files/training/config_nemotron3_nano_30b_a3b_fsdp2.yaml)
and the two Nemotron test configs are adopted from
`src/megatron/bridge/recipes/nemotronh/h100/nemotron_3_nano.py` and cross-checked against the model
report. No code was taken from Megatron-Bridge; it served as the configuration reference, and
`src/megatron/bridge/models/nemotronh/nemotron_h_bridge.py` documents the HF parameter mapping that
guided the module naming.

Files with **no** third-party derivation (written for Modalities, structured after its existing
GPT-2 components): `norms.py`, `nemotron_model_factory.py`, `nemotron_mfu.py`, and the
partitioning algorithm in `nemotron_stages_generator.py`.
