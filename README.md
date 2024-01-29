# Modalities


# Getting started
For training and evaluation a model, feel free to checkout [this](https://github.com/Modalities/modalities/blob/main/examples/getting_started/getting_started_example.md) getting started tutorial, in which we train a small, 60M-parameter GPT model on a tiny subset of the Redpajama V2 dataset. 
Also, see our WIki and API reference documentation: https://modalities.github.io/modalities/

# Installation

Create conda environment and activate it via 
```
conda create -n modalities python=3.10
conda activate modalities
```

then, install the repository via

```
pip install -e . 
```

If you want to contribute, have look at `CONTRIBUTING.md`.



# Usage
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

# Pydantic and ClassResolver

The mechanismn introduced to instantiate classes via `type_hint` in the `config.yaml`, utilizes 
1) Omegaconf to load the config yaml file
2) Pydantic for the validation of the config
3) ClassResolver to instantiate the correct, concrete class of a class hierarchy.

Firstly, Omegaconf loads the config yaml file and resolves internal refrences such as `${subconfig.attribue}`. 

Then, Pydantic validates the whole config as is and checks that each of the sub-configs are `pydantic.BaseModel` classes.
For configs, which allow different concrete classes to be instantiated by `ClassResolver`, the special member names `type_hint` and `config` are introduced.
With this we utilize Pydantics feature to auto-select a fitting type based on the keys in the config yaml file.

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
* a `type_hint` in a `BaseModel` config must be of type `modalities.config.lookup_types.LookupEnum` and 
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

To add a new resolver use `add_resolver`, and the corresponding added resolver will be accessible by the register_key given during adding.
For access use the `build_component_by_key_query` function of the `ResolverRegistry`.


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

# MemMap Datasets

## MemMapDataset Index Generator

The `MemMapDataset` requires an index file providing the necessary pointers into the raw data file. The `MemMapDataset` can create the index file lazily, however, it is advised to create it beforehand. This can be done by running

```sh
modalities create_memmap_index <path/to/jsonl/file>
```

The index will be created in the same directory as the raw data file. For further options you may look into the usage documentation via `modalities create_memmap_index --help`.

## Packed Dataset Generator

The `PackedMemMapDatasetContinuous` and `PackedMemMapDatasetMegatron` require a packed data file. To create the data file, you first have to generate a `MemMapDataset` index file as described [above](#memmapdataset-index-generator). Assuming the index and raw data are located in the same directory, you can simply execute the following command:

```sh
modalities create_packed_data <path/to/jsonl/file>
```

The packed data file will be created in the same directory as the raw data file. For further options you may look into the usage documentation via `modalities create_packed_data --help`.

### Packed Data Format

The packed data file is a bytestream containing both the tokenized data as well as an index denoting the start and length of the tokenized documents inside the bytestream. The data file consists of 3 concatenated parts:

header segment | data segment | index segment

* **header segment**: This section is a 8 bytes sized integer which encodes the length of the data segment in bytes.
* **data segment**: This section contains a concatenation of all documents in form of 4 bytes sized tokens. 
An end-of-sequence token is placed between consecutive documents.
* **index segment**: This section contains a pickled index which locates the documents inside the data segment.
 The index is basically a list of tuples, where each tuple contains the start position and length in bytes for the 
 corresponding document, e.g., `[(start_doc1, len_doc1), (start_doc2, len_doc2), ....]`.
