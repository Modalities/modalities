# LLMgym

# Installation

Create conda environment and activate it via 
```
conda create -n llm_gym python=3.10
conda activate llm_gym
```

then, install the repository via

```
cd src
pip install -e . 
```


To run the gpt2 training

```
cd src/gpt2
torchrun --nnodes 1 --nproc_per_node 4  main.py
```


VS code debugging config
```
        {
            "name": "GPT2 Example FSDP",  // accelerate launch --multi_gpu --gpu_ids "0,1,3" --num_processes=3 
            "type": "python",
            "request": "launch",
            //"module": "torchrun",
            "program": "/home/max-luebbering/miniconda3/envs/llm_gym/bin/torchrun",
            "console": "integratedTerminal",
            "env": {"CUDA_VISIBLE_DEVICES": "1,2,3,4,5,6,7"},
            "args": ["--nnodes", "1", "--nproc_per_node", "7",  "/raid/s3/opengptx/max_lue/LLMgym/src/llm_gym/fsdp.py"],
            "justMyCode": false
        }
```

# Usage
For running the training endpoint on multiple GPUs run `CUDA_VISIBLE_DEVICES=2,3 torchrun --nnodes 1 --nproc_per_node 2 --rdzv-endpoint=0.0.0.0:29502 src/llm_gym/__main__.py run --config_file_path config_files/config.yaml`.

Or, if you are a VsCode user, add this to your `launch.json`:
```json

        {
            "name": "Torchrun Main",
            "type": "python",
            "request": "launch",
            "module": "torch.distributed.run",
            "env": {
                "CUDA_VISIBLE_DEVICES": "2,3"
            },
            "args": [
                "--nnodes",
                "1",
                "--nproc_per_node",
                "2",
                "--rdzv-endpoint=0.0.0.0:29503",
                "src/llm_gym/__main__.py",
                "run",
                "--config_file_path",
                "config_files/config.yaml",
            ],
            "console": "integratedTerminal",
            "justMyCode": true,
            "envFile": "${workspaceFolder}/.env"
        }
```

# Pydantic and ClassResolver

The mechanismn introduced to instantiate classes via `type_hint` in the `config.yaml`, utilizes 
1) Omegaconf to load the config yaml file
2) Pydantic for the validation of the config
3) ClassResolver to instantiate the correct, concrete class of a class hierarchy.

Firstly, Omegaconf loads the config yaml file and resolves internal refrences such as `${subconfig.attribue}`. 

Then, Pydantic validates the whole config as is and checks that each of the sub-configs are `pydantic.BaseModel` classes.
For configs, which allow different concrete classes to be instantiated by `ClassResolver`, the special member names `type_hint` and `config` are introduced. With this we utilize Pydantics feature to auto-select a fitting type based on the keys in the config yaml file.

`ClassResolver` replaces large if-else control structures to infer the correct concrete type with a `type_hint` used for correct class selection:
```python
activation_resolver = ClassResolver(
    [nn.ReLU, nn.Tanh, nn.Hardtanh],
    base=nn.Module,
    default=nn.ReLU,
)
type_hint="ReLU"
activation_kwargs={...}
activation_resolver.make(type_hint, activation_kwargs),
```

In our implmentation we go a step further, as both,
* a `type_hint` in a `BaseModel` config must be of type `llm_gym.config.lookup_types.LookupEnum` and 
* `config` is a union of allowed concrete configs of base type `BaseModel`. 
`config` hereby replaces `activation_kwargs` in the example above, and replaces it with pydantic-validated `BaseModel` configs.

With this, a mapping between type hint strings needed for `class-resolver`, and the concrete class is introduced, while allowing pydantic to select the correct concrete config:

```python
from enum import Enum
from pydantic import BaseModel, PositiveInt, PositiveFloat, conint, confloat

class LookupEnum(Enum):
    @classmethod
    def _missing_(cls, value: str) -> type:
        """constructs Enum by member name, if not constructable by value"""
        return cls.__dict__[value]

class SchedulerTypes(LookupEnum):
    StepLR = torch.optim.lr_scheduler.StepLR
    ConstantLR = torch.optim.lr_scheduler.ConstantLR

class StepLRConfig(BaseModel):
    step_size: conint(ge=1)
    gamma: confloat(ge=0.0)


class ConstantLRConfig(BaseModel):
    factor: PositiveFloat
    total_iters: PositiveInt


class SchedulerConfig(BaseModel):
    type_hint: SchedulerTypes
    config: StepLRConfig | ConstantLRConfig
```

To allow a user-friendly instantiation, all class resolvers are defined in the `ResolverRegistry` and `build_component_by_config` as convenience function is introduced. Dependecies can be passed-through with the `extra_kwargs` argument:
```python
resolvers = ResolverRegister(config=config)
optimizer = ...  # our example dependency
scheduler = resolvers.build_component_by_config(config=config.scheduler, extra_kwargs=dict(optimizer=optimizer))
```

To add a new resolver use `add_resolver`, and the corresponding added resolver will be accessible by the register_key given during adding. For access use the `build_component_by_key_query` function of the `ResolverRegistry`.

# MemMapDataset Index Generator

The `MemMapDataset` requires an index file providing the necessary pointers into the raw data file. The `MemMapDataset` can create the index file lazyly, however, it is adviced to create it beforhand. This can be done by running

```sh
llm_gym create_mmap_index <path/to/jsonl/file>
```

The index will be created in the same directory as the raw data file.