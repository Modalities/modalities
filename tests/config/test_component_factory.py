from pathlib import Path
from typing import Dict, Union

from omegaconf import OmegaConf

from modalities.config.component_factory import ComponentFactory
from tests.config.hierarchical_instantiation.configs import (
    CompVConfig,
    CompWConfig,
    CompXConfig,
    CompYConfig,
    CompZConfig,
    ReferenceConfig,
)


def load_app_config_dict(config_file_path: Path) -> Dict:
    cfg = OmegaConf.load(config_file_path)
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    return config_dict


config_file_path = Path("config.yaml")
comp_config_types = Union[CompVConfig, CompWConfig, CompXConfig, CompYConfig, CompZConfig, ReferenceConfig]
component_names = ["comp_z_1", "comp_x_1", "comp_y_1"]

config_dict = load_app_config_dict(config_file_path=config_file_path)

components = ComponentFactory.build_config(
    config_dict=config_dict, config_types=comp_config_types, component_names=component_names
)
print(components)
