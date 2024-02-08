from pathlib import Path
from typing import Any, Dict, List, Union

from omegaconf import OmegaConf
from pydantic import RootModel

from modalities.config.hierarchical_dependency_injection.configs import (
    CompConfigABC,
    CompVConfig,
    CompWConfig,
    CompXConfig,
    CompYConfig,
    CompZConfig,
)


class ComponentFactory:
    @staticmethod
    def build_config(config_file_path: Path, config_types, component_names: List[str]) -> Dict[str, Any]:
        config_dict = ComponentFactory._get_config(file_path=config_file_path)
        component_dict = ComponentFactory._build_components(
            config_dict=config_dict, config_types=config_types, component_names=component_names
        )
        return component_dict

    @staticmethod
    def _get_config(file_path: Path) -> Dict:
        cfg = OmegaConf.load(file_path)
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        return config_dict

    @staticmethod
    def _build_components(config_dict: Dict, config_types, component_names: List[str]) -> Dict[str, Any]:
        components = {
            name: ComponentFactory._build_component(component_config_dict, config_types)
            for name, component_config_dict in config_dict.items()
        }
        return {name: components[name] for name in component_names}

    @staticmethod
    def _build_component(component_config: Union[Dict, List, Any], config_types) -> Any:
        # build sub components first via recursion
        if isinstance(component_config, dict):
            materialized_component_config = {}
            for sub_component_name, sub_component_config_dict in component_config.items():
                materialized_component_config[sub_component_name] = ComponentFactory._build_component(
                    sub_component_config_dict, config_types
                )

            # if the config is component_config then we instantiate the component
            if ComponentFactory._is_component_config(config_dict=component_config):
                # instantiate component config
                component_config = ComponentFactory._instantiate_component_config(
                    config_dict=materialized_component_config, config_types=config_types
                )
                # instantiate component
                component = ComponentFactory._instantiate_component(component_config=component_config)
                return component
            return materialized_component_config

        elif isinstance(component_config, list):
            materialized_component_config = []
            for sub_component_config in component_config:
                materialized_component_config.append(
                    ComponentFactory._build_component(sub_component_config, config_types)
                )
            return materialized_component_config

        else:
            return component_config

    @staticmethod
    def _is_component_config(config_dict: Dict) -> bool:
        return "type_hint" in config_dict.keys()

    @staticmethod
    def _instantiate_component_config(config_dict: Dict, config_types) -> CompConfigABC:
        comp_config = RootModel[config_types].model_validate(config_dict, strict=True).root
        return comp_config

    @staticmethod
    def _instantiate_component(component_config: CompConfigABC) -> Any:
        component_type = component_config.type_hint.value
        component = component_type(**component_config.model_dump(exclude=["type_hint"]))
        return component


if __name__ == "__main__":
    config_file_path = Path(
        "/raid/s3/opengptx/max_lue/modalities/src/modalities/config/hierarchical_dependency_injection/config.yaml"
    )
    comp_config_types = Union[CompVConfig, CompWConfig, CompXConfig, CompYConfig, CompZConfig]
    component_names = ["comp_z_1", "comp_x_1", "comp_y_1"]
    components = ComponentFactory.build_config(
        config_file_path=config_file_path, config_types=comp_config_types, component_names=component_names
    )
    print(components)
