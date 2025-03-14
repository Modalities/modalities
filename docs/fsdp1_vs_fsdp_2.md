# Instantiation dependencies (Training)

## FSDP1

Firstly, we instantiate the model (random weights) on each rank, initialize the weights following a pre-defined distribution, and then wrap the model with FSDP1. During wrapping the weights on rank 0 are sharded to the other ranks, thus initialisation on the other ranks is being replaced.

```
model_raw -> initialized_model -> wrapped_model
```

#### LR Scheduler requirements
- optimizer  [Reference](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)

#### Optimizer requirements
- wrapped_model (fsdp1 wrapped)
[Reference](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)

#### Gradient clipper requirements
- wrapped_model (fsdp1 wrapped) [Reference](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.clip_grad_norm_)

#### AppState requirements
- wrapped_model (fsdp1 wrapped)
- optimizer
- lr_scheduler

## FSDP2
In the case of FSDP2, we first instantiate the model, wrap it with FSDP2 and then initialize the weights following a pre-defined distribution. The model_raw can be instantiated on the a meta device. After sharding with FSDP2 (fsdp_model), the model is still on a meta device and only during the initialisation, the model is moved to the data device (see [here](https://github.com/Modalities/modalities/blob/a7f683a6202209dcf8e78cbff4ca68991e89e081/src/modalities/models/model_factory.py)).
The initialisation happens in place on the DTensors, similar to how it is done in [torch titan](https://github.com/pytorch/torchtitan/blob/b291ad662493b63d25b038a30a915082d3617baf/torchtitan/models/llama/model.py).

```
model_raw -> fsdp_model -> initialized_model
```

#### LR Scheduler requirements
- optimizer [Reference](https://github.com/pytorch/torchtitan/blob/b291ad662493b63d25b038a30a915082d3617baf/torchtitan/train.py#L206-L207)

#### Optimizer requirements
- initialized_model (fsdp2 wrapped and weight initialized) [Reference](https://github.com/pytorch/torchtitan/blob/b291ad662493b63d25b038a30a915082d3617baf/torchtitan/train.py#L206-L207)

#### Gradient clipper requirements
- initialized_model (fsdp2 wrapped and weight initialized) [Reference](https://github.com/pytorch/torchtitan/blob/b291ad662493b63d25b038a30a915082d3617baf/torchtitan/train.py#L336)

#### AppState requirements
- initialized_model (fsdp2 wrapped and weight initialized)
- optimizer
- lr_scheduler


## Difference between FSDP1 and FSDP2
The main difference in the instantiation dependencies between FSDP1 and FSDP2 is that in FSDP1, the model is wrapped after the weights are initialized, while in FSDP2, the model is wrapped before the weights are initialized. Therefore, FSDP1 requires the model to be fully materialized in CPU RAM before wrapping, as otherwise the weights cannot be initialized. In the case of FSDP2, we can have the model on a meta device, shard it and then move it to the data device during the initialization. This way, only the shards are materialized in GPU RAM and we have an extremely low CPU RAM footprint. 
See also the torch titan [documentation](https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md) on the differences between FSDP1 and FSDP2. 


# Instantiation dependencies (Warmstart)

## FSDP1
Similar to before, we first instantiate the model (random weights), initialize the weights following a pre-defined distribution, and load the actual weights from the checkpoint and shard the model with FSDP1. Loading and sharding is one step in modalities. We first load the model into the CPU RAM on rank 0 and then wrap the model with FSDP1, sharding the model to the other ranks.
    
https://github.com/Modalities/modalities/blob/a7f683a6202209dcf8e78cbff4ca68991e89e081/src/modalities/checkpointing/fsdp/fsdp_checkpoint_loading.py#L58-L61.

The optimizer is initialized as before with the wrapped_model and the optimizer states are subsequently loaded. 

```
model_raw -> initialized_model -> wrapped_model (checkpoint loaded and fsdp-wrapped)
optimizer_original -> optimizer (checkpoint loaded)
```

#### LR Scheduler requirements
- optimizer (checkpoint loaded)

### Optimizer (initial) requirements
- wrapped_model (fsdp1 wrapped) [Reference](https://github.com/lessw2020/transformer_framework/blob/main/model_checkpointing/load_save_full.ipynb)

#### Optimizer (checkpoint loaded) requirements
- optimizer_inital (optimizer, same as for training)
- wrapped_model (fsdp1 wrapped) [Reference](https://github.com/lessw2020/transformer_framework/blob/main/model_checkpointing/load_save_full.ipynb)

#### Gradient clipper requirements
- wrapped_model (fsdp1 wrapped and checkpoint loaded) [Reference](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.clip_grad_norm_)

#### AppState requirements
- wrapped_model (fsdp1 wrapped and checkpoint loaded)
- optimizer (checkpoint loaded)
- lr_scheduler


## FSDP2

Firstly, the model_raw is created on the meta_device, sharded with FSDP2 and then the model is initialized with the weights by sampling from the specified distribution. Finally, the actual weights from the checkpoint are loaded during app_state instantiation.


```
model_raw -> fsdp_model -> initialized_model
```


#### LR Scheduler requirements
- optimizer (does not contain checkpoint!) [Reference](https://github.com/pytorch/torchtitan/blob/b291ad662493b63d25b038a30a915082d3617baf/torchtitan/train.py#L207)

#### Optimizer requirements
- initialized_model (init + sharded but does not contain checkpoint!) [Reference](https://github.com/pytorch/torchtitan/blob/b291ad662493b63d25b038a30a915082d3617baf/torchtitan/train.py#L206)

#### Gradient clipper requirements
- initialized_model ((init + sharded but does not contain checkpoint!)

We pass in the reference to the initialized model even though torch titan uses the checkpointed one (from the app_state). Nevertheless, this should not make a difference since the reference to the intialized and checkpointed model should be the same and the gradient clipper is stateless.

#### AppState (raw) requirements
- initialized_model ((init + sharded but does not contain checkpoint!)
- optimizer
- lr_scheduler


#### AppState (checkpoint loaded) requirements
- raw_app_state (contains initialized_model, optimizer, lr_scheduler) [Reference 1](https://github.com/pytorch/torchtitan/blob/b291ad662493b63d25b038a30a915082d3617baf/torchtitan/components/checkpoint.py#L429) and [Reference 2](https://github.com/pytorch/torchtitan/blob/b291ad662493b63d25b038a30a915082d3617baf/torchtitan/components/checkpoint.py#L289)
