from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

from omegaconf import OmegaConf
from pydantic import BaseModel, RootModel

from modalities.config.hierarchical_instantiation_trial.configs import (
    CompWConfig,
    CompXConfig,
    CompYConfig,
    CompZConfig,
    ReferenceConfig,
)

component_config_types: List[BaseModel] = [CompWConfig, CompXConfig, CompYConfig, CompZConfig]


class AppConfig(RootModel):
    root: Dict[str, Union[CompWConfig, CompXConfig, CompYConfig, CompZConfig, ReferenceConfig]]

    # Not possibley yet to forbid extra arguments, yet.
    # As of now they are by default being ignored.
    # see: https://docs.pydantic.dev/latest/errors/usage_errors/#root-model-extra
    # model_config = {'extra': 'allow'}

    # # Custom validator to check each item in the dictionary
    # @validator("root", pre=True, each_item=True)
    # def _validate_app_config(v, values, **kwargs):
    #     if isinstance(v, dict):
    #         # Attempt to parse the dictionary into one of the pet models
    #         # concrete_component_config_types = [ComponentConfig[cct] for cct in component_config_types]
    #         for base_config_type in [ComponentConfig, ReferenceConfig]:
    #             try:
    #                 return base_config_type(**v)
    #             except ValueError:
    #                 continue
    #         raise ValueError(f"Value does not match any config model: {v}")
    #     return v

    # To interact with the model as a dictionary
    def __getitem__(self, item):
        return self.root[item]

    def __setitem__(self, key, value):
        self.root[key] = value

    def __iter__(self):
        return iter(self.root)

    def keys(self):
        return self.root.keys()

    def values(self):
        return self.root.values()

    def items(self):
        return self.root.items()


def load_app_config_dict(config_file_path: Path) -> Dict:
    cfg = OmegaConf.load(config_file_path)
    return OmegaConf.to_container(cfg, resolve=True)


if __name__ == "__main__":
    config_dict = load_app_config_dict(
        config_file_path=Path(
            "/raid/s3/opengptx/max_lue/modalities/src/modalities/config/hierarchical_instantiation_trial/config.yaml"
        )
    )

    comp_config_types = Union[CompWConfig, CompXConfig, CompYConfig, CompZConfig, ReferenceConfig]
    comp_config_type_union = Union[
        CompWConfig[comp_config_types],
        CompXConfig[comp_config_types],
        CompYConfig[comp_config_types],
        CompZConfig[comp_config_types],
        ReferenceConfig,
    ]

    config = RootModel[Dict[str, Any]].model_validate(config_dict, strict=True)

    # config = AppConfig.model_validate(config_dict, strict=True)
    print(config)


# The config dependencies should contain the actual component types
# and the config types.
# Idea: Define the configs with the dependency IFs.
# We traverse the config dictionary and each component gets build.
# We track the components by building up a second dependency graph with
# with the actual components and not the configs.
# When instantiating we first create the component dictionary with the config
# values and the instantiated dependency components. We run the dictionry
# through Pydantic and feed in the config to the component type that is to
# be instantiated.
